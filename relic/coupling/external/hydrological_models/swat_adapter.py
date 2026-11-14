"""
SWAT Model Data Adapter

This module implements the SWATAdapter for accessing and processing data from 
the Soil and Water Assessment Tool (SWAT) hydrological model output files.
It handles SWAT-specific variable naming, file formats, and watershed structure.

The adapter supports standard SWAT output formats including text files and
databases, and follows CF-1.8 conventions where possible.

References:
    - SWAT Documentation: https://swat.tamu.edu/
    - SWAT I/O Documentation: https://swat.tamu.edu/software/swat-input-output-file-documentation/
    - CF Conventions 1.8: http://cfconventions.org/cf-conventions/v1.8/
    
Mathematical Expressions:
    Water balance: P - ET - SW_out - Perc - Revap = DeltaSW
    Surface runoff: SURQ = CN_adj * (P - I_a)^2/(P - I_a + S)
    Sediment yield: SED = 11.8 * (Q_surf * q_peak * A_hru)^0.56 * K * C * P * LS * CFRG
    Nutrient transport: N_surf = 0.78 * SED * (CONC_N / 1000000)
    Crop growth: BIO = HI * (1 - exp(-k * LAI)) * PAR
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import re
from dataclasses import dataclass

import torch
import xarray as xr
import numpy as np
from ..common.base_adapter import validate_record_interval, validate_time_interpolation
from ..common import (
    ExternalAdapter, DataSource,
    TimeWindow, SpatialWindow,
    TimeHandler, RegridOperator,
    IOHandler
)
# ProcessCategory removed - ExternalAdapter handles category internally

# Parameter management imports

logger = logging.getLogger(__name__)

# SWAT 2012 print settings of a run as named in file.cio and readfile.f90 (Arnold et al., 2012,
# SWAT Input/Output Documentation, file.cio): NBYR number of simulated years, IYR beginning year,
# IDAF beginning julian day, IDAL ending julian day of the last simulated year, IPRINT print code
# (0 monthly, 1 daily, 2 annual records), NYSKIP number of warm-up years not printed, ICALEN date
# code of daily rows (0 julian day, 1 month/day/year). Domains: the documented codes,
# four-digit years (output.hru prints the year as i4) and julian days, where IDAF = 0 and
# IDAL = 0 mean the first and the last day of the year (simulate.f90).
_SWAT_PRINT_SETTING_DOMAINS = {
    'NBYR': (1, 8999),
    'IYR': (1000, 9998),
    'IDAF': (0, 366),
    'IDAL': (0, 366),
    'IPRINT': (0, 2),
    'NYSKIP': (0, 8999),
    'ICALEN': (0, 1),
}


def _validate_swat_print_setting(name: str, value: Any, source: str) -> None:
    """Raise unless ``value`` is an integer inside the documented domain of SWAT setting ``name``."""
    low, high = _SWAT_PRINT_SETTING_DOMAINS[name]
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"SWAT {name} ({source}) must be an integer in [{low}, {high}], got {value!r}.")


@dataclass
class SWATConfig:
    """Configuration for SWAT adapter"""
    root_path: Union[str, Path]
    # Output table read: only "hru" (the SWAT 2012 output.hru text layout) is implemented.
    output_level: str
    watershed_file: Optional[Union[str, Path]] = None  # Watershed shapefile for spatial info
    # Output record interval declared for water depths per record when the file has no
    # CF time bounds (e.g. 1.0 and "day" for a daily output step); never inferred. SWAT text
    # outputs take their record bounds from the print settings below and reject it.
    record_interval: Optional[float] = None
    record_interval_units: Optional[str] = None
    # Print settings of the run (file.cio names in capitals): nbyr NBYR, iyr IYR, idaf IDAF, idal
    # IDAL, iprint IPRINT, nyskip NYSKIP, icalen ICALEN, and the CF calendar of the run (file.cio
    # declares none). A setting read from root_path/file.cio is used; a different configured
    # value raises; a setting declared by neither raises when a text output is read.
    nbyr: Optional[int] = None
    iyr: Optional[int] = None
    idaf: Optional[int] = None
    idal: Optional[int] = None
    iprint: Optional[int] = None
    nyskip: Optional[int] = None
    icalen: Optional[int] = None
    calendar: Optional[str] = None

    # Rule of a forcing read at the simulation time: 'lower' (t_k <= t < t_(k+1) holds record k)
    # or 'linear' (between the two records around t); required by forward, never assumed.
    time_interpolation: Optional[str] = None
    # Catalog variables the adapter writes (its declared outputs; each a mapping target of the
    # source in InternalSchema); None declares every mapping target of the source.
    provided_variables: Optional[List[str]] = None

    def __post_init__(self) -> None:
        validate_time_interpolation(self.time_interpolation)
        validate_record_interval(self.record_interval, self.record_interval_units)
        if self.output_level != "hru":
            raise ValueError(
                f"SWAT output_level {self.output_level!r}: only the SWAT 2012 output.hru text layout is "
                "read (output_level='hru'); output.sub, output.rch, output.rsv and the SQLite outputs "
                "have no verified layout here."
            )
        for name in _SWAT_PRINT_SETTING_DOMAINS:
            value = getattr(self, name.lower())
            if value is not None:
                _validate_swat_print_setting(name, value, f"SWATConfig.{name.lower()}")
        if self.calendar is not None and (not isinstance(self.calendar, str) or not self.calendar.strip()):
            raise ValueError(f"SWATConfig.calendar must be a non-empty CF calendar name, got {self.calendar!r}.")


class SWATAdapter(ExternalAdapter):
    """
    Adapter for Soil and Water Assessment Tool (SWAT) hydrological model output.
    
    This adapter provides access to SWAT model output files, handling the
    model's specific output formats, watershed structure, and variable naming
    conventions. It can process outputs at different spatial scales.
    
    Key Features:
    - Multiple output level support (HRU, subbasin, reach, reservoir)
    - Text file and database format handling
    - Watershed spatial organization
    - Water quality and quantity processing
    - CF-1.8 compliant metadata generation
    
    Attributes:
        config: SWAT configuration parameters
        discovered_files: List of discovered SWAT files
        current_datasets: Currently loaded datasets
        watershed_info: Spatial watershed information
        variable_definitions: SWAT variable definitions
    """
    
    # SWAT 2012 output.hru (Arnold et al., 2012, SWAT Input/Output Documentation, section 32.3;
    # SWAT 2012 rev 670 source). headout.f90 format 1020 writes the header
    # 'LULC  HRU       GIS  SUB  MGT  MON' + '   AREAkm2' + one a10 field per variable (the heds
    # name followed by a unit suffix, the file's declaration of the column unit; a field without
    # a suffix declares no unit); format 1021 (ICALEN=1) writes 'MO DA   YR' instead of MON. Rows:
    # hrumon.f90 and hruyr.f90 format 1000 (a4,i5,1x,a5,a4,i5,1x,i4,1x,i4,e10.5,66f10.3,1x,e10.5,
    # 1x,e10.5,...), hruyr.f90 and hruaa.f90 format 1001 with HRU as i7, hruaa.f90 MON as f4.1
    # (years averaged), format 2000 appending the year as (1x,i4). The row layouts of
    # output.sub/.rch/.rsv are not implemented.
    TEXT_OUTPUT_FILES = {
        'output.hru': ('hru', 'HRU'),
    }
    TEXT_HEADER_LAYOUTS = {
        0: 'LULC  HRU       GIS  SUB  MGT  MON   AREAkm2',
        1: 'LULC  HRU       GIS  SUB  MGT MO DA   YR   AREAkm2',
    }
    TEXT_FIELD_WIDTH = 10
    TEXT_ROW_HRU_WIDTHS = (5, 7)
    # 0-based value fields preceded by a 1x separator (66f10.3,1x,e10.5,1x,e10.5,...).
    TEXT_ROW_SPACED_VALUES = (66, 67)
    HEADER_UNIT_SUFFIXES = (
        ('MJ/m2', 'MJ m-2'),
        ('kg/ha', 'kg ha-1'),
        ('t/ha', 't ha-1'),
        ('km2', 'km2'),
        ('dgC', 'degC'),
        ('cms', 'm3 s-1'),
        ('mm', 'mm'),
    )
    TEXT_RECORD_COLUMN = 'MON'
    PRINT_FREQUENCIES = {0: 'monthly', 1: 'daily', 2: 'annual'}
    # Largest MON of a period block: the month (IPRINT=0) or the julian day (IPRINT=1); an
    # annual print (IPRINT=2) has no period blocks.
    PERIOD_MON_LIMITS = {0: 12, 1: 366, 2: 0}
    MON_RULES = {
        0: 'the month (1-12) of a monthly block, the four-digit year of an annual block or the '
           'years averaged (f4.1) of the average-annual block',
        1: 'the julian day (1-366) of a daily block',
        2: 'the four-digit year of an annual block or the years averaged (f4.1) of the '
           'average-annual block',
    }
    # SWAT 2012 rev 670 simulate.f90 sets a leap year when mod(iyr, 4) == 0; the CF julian
    # calendar has this rule and the same month lengths, so a declared calendar must give the
    # year and month lengths of the julian calendar for every simulated year.
    SWAT_CALENDAR = 'julian'
    # References: SWAT 2012 rev 670 readfile.f90 read statements of unit 101 (file.cio) at the
    #   lines listed below, formats 5000 '(6a)', 5100 '(20a4)' and 5101 '(a80)' (lines 777-779),
    #   character(len=13) file names and character(len=4) title(60) (modparm.f90); zero0.f90
    #   sets icalen = 0 and runs from allocate_parms.f90 before readfile (main.f90); Fortran 2008
    #   (ISO/IEC 1539-1:2010) 10.4 (format reversion starts a new record) and 10.10.3
    #   (list-directed input).
    # Mathematical expression: a formatted read of k items under a format with m data edit
    #   descriptors consumes ceil(k / m) records; a list-directed read of n values consumes the
    #   records up to the one holding its n-th value (blank records give no value, r*c gives r
    #   values, a slash ends the read) and ignores the rest of that record.
    # Parameter meanings: k items of the input list (60 for title, 18 for rfile and tfile, 1
    #   otherwise); m = 1 for a80, 20 for 20a4, 6 for 6a; n values of a list-directed read (20
    #   for ipdvar, ipdvas and ipdhru, 15 for ipdvab, 1 otherwise).
    # Units: records (file lines) and values (counts); the settings read are dimensionless codes,
    #   years (NBYR, IYR, NYSKIP) and julian days (IDAF, IDAL).
    # Domain/constraints: reads without iostat must find their records, so a shorter file
    #   raises; reads with iostat=eof (readfile.f90 line 474 on) stop at the end of the file and
    #   the variables keep their initial values (ICALEN 0); a setting's first value is an
    #   integer constant; a writer label ('value | NAME : description') on a setting record must
    #   name that setting and a setting label on any other record raises.
    # Sign convention: not applicable; the file position only advances.
    # Entries: (readfile.f90 line, edit, items, variable, iostat=eof).
    FILE_CIO_READS = (
        (151, 'a80', 1, 'titldum', False),
        (152, 'a80', 1, 'titldum', False),
        (153, '20a4', 60, 'title', False),
        (156, 'a80', 1, 'titldum', False),
        (157, '6a', 1, 'figfile', False),
        (158, 'list', 1, 'nbyr', False),
        (159, 'list', 1, 'iyr', False),
        (160, 'list', 1, 'idaf', False),
        (161, 'list', 1, 'idal', False),
        (167, 'a80', 1, 'titldum', False),
        (168, 'list', 1, 'igen', False),
        (169, 'list', 1, 'pcpsim', False),
        (170, 'list', 1, 'idt', False),
        (171, 'list', 1, 'idist', False),
        (172, 'list', 1, 'rexp', False),
        (173, 'list', 1, 'nrgage', False),
        (174, 'list', 1, 'nrtot', False),
        (175, 'list', 1, 'nrgfil', False),
        (176, 'list', 1, 'tmpsim', False),
        (177, 'list', 1, 'ntgage', False),
        (178, 'list', 1, 'nttot', False),
        (179, 'list', 1, 'ntgfil', False),
        (180, 'list', 1, 'slrsim', False),
        (181, 'list', 1, 'nstot', False),
        (182, 'list', 1, 'rhsim', False),
        (183, 'list', 1, 'nhtot', False),
        (184, 'list', 1, 'wndsim', False),
        (185, 'list', 1, 'nwtot', False),
        (186, 'list', 1, 'fcstyr', False),
        (187, 'list', 1, 'fcstday', False),
        (188, 'list', 1, 'fcstcycles', False),
        (189, 'a80', 1, 'titldum', False),
        (190, '6a', 18, 'rfile', False),
        (191, 'a80', 1, 'titldum', False),
        (192, '6a', 18, 'tfile', False),
        (193, '6a', 1, 'slrfile', False),
        (194, '6a', 1, 'rhfile', False),
        (195, '6a', 1, 'wndfile', False),
        (196, '6a', 1, 'fcstfile', False),
        (256, 'a80', 1, 'titldum', False),
        (257, '6a', 1, 'bsnfile', False),
        (263, 'a80', 1, 'titldum', False),
        (264, '6a', 1, 'plantdb', False),
        (265, '6a', 1, 'tilldb', False),
        (266, '6a', 1, 'pestdb', False),
        (267, '6a', 1, 'fertdb', False),
        (268, '6a', 1, 'urbandb', False),
        (283, 'a80', 1, 'titldum', False),
        (284, 'list', 1, 'isproj', False),
        (285, 'list', 1, 'iclb', False),
        (286, '6a', 1, 'calfile', False),
        (289, 'a80', 1, 'titldum', False),
        (290, 'list', 1, 'iprint', False),
        (291, 'list', 1, 'nyskip', False),
        (295, 'list', 1, 'ilog', False),
        (296, 'list', 1, 'iprp', False),
        (297, 'a80', 1, 'titldum', False),
        (307, 'a80', 1, 'titldum', False),
        (308, 'list', 20, 'ipdvar', False),
        (363, 'a80', 1, 'titldum', False),
        (364, 'list', 15, 'ipdvab', False),
        (385, 'a80', 1, 'titldum', False),
        (386, 'list', 20, 'ipdvas', False),
        (469, 'a80', 1, 'titldum', False),
        (470, 'list', 20, 'ipdhru', False),
        (474, 'a80', 1, 'titldum', True),
        (476, '6a', 1, 'atmofile', True),
        (485, 'list', 1, 'iphr', True),
        (490, 'list', 1, 'isto', True),
        (496, 'list', 1, 'isol', True),
        (508, 'list', 1, 'i_subhw', True),
        (513, '6a', 1, 'septdb', True),
        (526, 'list', 1, 'ia_b', True),
        (531, 'list', 1, 'ihumus', True),
        (537, 'list', 1, 'itemp', True),
        (541, 'list', 1, 'isnow', True),
        (695, 'list', 1, 'imgt', True),
        (723, 'list', 1, 'iwtr', True),
        (740, 'list', 1, 'icalen', True),
    )
    # Data edit descriptors per record of the formatted reads (formats 5101, 5100, 5000).
    FILE_CIO_FORMAT_ITEMS = {'a80': 1, '20a4': 20, '6a': 6}
    # Initial values of settings read with iostat=eof (zero0.f90).
    FILE_CIO_INITIAL_VALUES = {'ICALEN': 0}
    FILE_CIO_LABEL = re.compile(r"\|\s*([A-Za-z_][A-Za-z0-9_]*)")
    FORTRAN_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?")

    def __init__(
        self,
        config: Optional[SWATConfig] = None,
        time_window: Optional[TimeWindow] = None,
        spatial_window: Optional[SpatialWindow] = None,
        **kwargs
    ):
        """
        Initialize SWAT adapter.
        
        Parameters:
            config: SWAT configuration
            time_window: Temporal subset window 
            spatial_window: Spatial subset window
            **kwargs: Additional arguments passed to ExternalAdapter
        """
        self._require_adapter_config(config, SWATConfig, "SWATAdapter")
        if config is None:
            raise ValueError(
                "SWATAdapter needs a SWATConfig declaring root_path and output_level ('hru'); the SWAT "
                "output directory is never assumed."
            )
        super().__init__(
            source=DataSource.SWAT,
            name="SWATAdapter",
            model_id="swat",
            provided_variables=config.provided_variables,
            **kwargs
        )

        self.config = config
        self.time_window = time_window
        self.spatial_window = spatial_window
        
        # Internal state
        self.discovered_files: List[Path] = []
        self.current_datasets: Dict[str, xr.Dataset] = {}
        self.watershed_info: Dict[str, Any] = {}
        self.variable_definitions: Dict[str, Dict] = {}
        
        # Initialize handlers
        self.time_handler = TimeHandler()
        self.regrid_operator = RegridOperator()
        self.io_handler = IOHandler()
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Load SWAT variable definitions
        self._load_swat_variable_definitions()
        
    def _load_swat_variable_definitions(self):
        """Load standard SWAT variable definitions"""
        self.variable_definitions = {
            # Hydrology variables
            'PRECIP': {
                'long_name': 'Precipitation',
                'units': 'mm'
            },
            'SURQ_GEN': {
                'long_name': 'Surface Runoff',
                'units': 'mm'
            },
            'LATQGEN': {
                'long_name': 'Lateral Flow',
                'units': 'mm'
            },
            'GW_Q': {
                'long_name': 'Groundwater Flow',
                'units': 'mm'
            },
            'WYLD': {
                'long_name': 'Water Yield',
                'units': 'mm'
            },
            'ET': {
                'long_name': 'Evapotranspiration',
                'units': 'mm'
            },
            # Profile water above the wilting point; no CF standard name describes it.
            'SW': {
                'long_name': 'Soil Water Content Above Wilting Point (Profile)',
                'units': 'mm'
            },
            'PERC': {
                'long_name': 'Percolation',
                'units': 'mm'
            },
            
            # Sediment variables
            'SYLD': {
                'long_name': 'Sediment Yield',
                'units': 't/ha'
            },
            'USLE': {
                'long_name': 'USLE Soil Loss',
                'units': 't/ha'
            },
            
            # Nutrient variables  
            'ORGN': {
                'long_name': 'Organic Nitrogen Loss',
                'units': 'kg N/ha'
            },
            'ORGP': {
                'long_name': 'Organic Phosphorus Loss',
                'units': 'kg P/ha'
            },
            'NSurQ': {
                'long_name': 'Nitrogen in Surface Runoff',
                'units': 'kg N/ha'
            },
            'PSurQ': {
                'long_name': 'Phosphorus in Surface Runoff',
                'units': 'kg P/ha'
            },
            'NLatQ': {
                'long_name': 'Nitrogen in Lateral Flow',
                'units': 'kg N/ha'
            },
            'PLatQ': {
                'long_name': 'Phosphorus in Lateral Flow',
                'units': 'kg P/ha'
            },
            
            # Management variables
            'BIOM': {
                'long_name': 'Biomass',
                'units': 't/ha'
            },
            'LAI': {
                'long_name': 'Leaf Area Index',
                'units': 'm2/m2'
            },
            'YLD': {
                'long_name': 'Crop Yield',
                'units': 't/ha'
            }
        }
        
    def discover(
        self, 
        variables: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Discover available SWAT output files.
        
        Scans the specified directory for SWAT output files based on the
        configuration. Identifies output files by naming convention and
        analyzes their structure.
        
        Parameters:
            variables: List of SWAT variable names to check for
            **kwargs: Additional discovery parameters
            
        Returns:
            Dictionary containing discovered file information and metadata
            
        Raises:
            FileNotFoundError: If root path does not exist
            ValueError: If no matching files found
        """
        self.logger.info(f"Discovering SWAT files in {self.config.root_path}")
        
        root_path = Path(self.config.root_path)
        if not root_path.exists():
            raise FileNotFoundError(f"SWAT root path does not exist: {root_path}")
        
        patterns = self._get_swat_file_patterns()
        
        matching_files = []
        for pattern in patterns:
            matching_files.extend(root_path.glob(pattern))
            
        if not matching_files:
            raise ValueError(f"No SWAT files found with patterns: {patterns}")
        
        self.discovered_files = matching_files
        
        # Analyze file contents
        file_analysis = self._analyze_swat_files(matching_files, variables)
        
        results = {
            "source": "SWAT",
            "total_files": len(matching_files),
            "output_format": "text",
            "output_level": file_analysis["level"],
            "time_coverage": file_analysis["time_coverage"],
            "available_variables": file_analysis["variables"],
            "spatial_info": file_analysis["spatial_info"]
        }
        
        self.watershed_info = file_analysis["spatial_info"]
        
        self.logger.info(
            f"Discovered {results['total_files']} SWAT files at "
            f"{results['output_level']} level"
        )
        
        return results
    
    def _get_swat_file_patterns(self) -> List[str]:
        """SWAT text output file names of the configured output level (output.hru only)."""
        return [name for name, (level, _key) in self.TEXT_OUTPUT_FILES.items() if level == self.config.output_level]

    def _analyze_swat_files(
        self, 
        files: List[Path], 
        variables: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze SWAT files to extract metadata"""
        
        output_level = self.config.output_level
        all_variables = set()
        time_range = {'start': None, 'end': None}

        # Text output headers (errors propagate)
        for filepath in files:
            _output_type, _key = self._text_output_type(filepath)
            _row, columns = self._parse_swat_text_header(filepath)
            all_variables.update(name for name, _units in columns)

        # Basic spatial info (would need watershed file for detailed info)
        spatial_info = {
            'type': 'watershed',
            'level': output_level,
            'projection': 'unknown'  # Would need to read from watershed shapefile
        }
        
        return {
            'format': 'text',
            'level': output_level,
            'variables': sorted(list(all_variables)),
            'time_coverage': time_range,
            'spatial_info': spatial_info
        }
    
    def open(self, **kwargs) -> Dict[str, Any]:
        """
        Open and load discovered SWAT datasets.
        
        Opens SWAT files and processes them according to configuration.
        Handles different output formats and levels.
        
        Parameters:
            **kwargs: Additional opening parameters
            
        Returns:
            Dictionary containing opened datasets and metadata
            
        Raises:
            ValueError: If no files have been discovered
            IOError: If files cannot be opened
        """
        if not self.discovered_files:
            raise ValueError("No SWAT files discovered. Call discover() first.")
        
        self.logger.info(f"Opening {len(self.discovered_files)} SWAT files")
        
        try:
            datasets = self._open_text_files()
            
            # Apply temporal and spatial subsetting to all datasets
            for name, dataset in datasets.items():
                datasets[name] = self._apply_subsetting(dataset)
                datasets[name] = self._add_swat_metadata(datasets[name])
            
            self.current_datasets = datasets
            
            return {
                "source": "SWAT",
                "datasets": list(datasets.keys()),
                "temporal_coverage": self._get_temporal_coverage(),
                "variables": self._get_all_variables()
            }
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            self.logger.error(f"Failed to open SWAT datasets: {e}")
            raise IOError(f"Could not open SWAT data: {e}")
    
    def _open_text_files(self) -> Dict[str, xr.Dataset]:
        """Open SWAT 2012 text outputs (output.hru/.sub/.rch/.rsv); read errors propagate."""
        datasets: Dict[str, xr.Dataset] = {}
        for filepath in self.discovered_files:
            output_type, _key = self._text_output_type(filepath)
            if output_type in datasets:
                raise ValueError(f"Two SWAT {output_type} outputs were discovered ({filepath.name}).")
            datasets[output_type] = self._read_swat_text_file(filepath)
        return datasets

    def _text_output_type(self, filepath: Path) -> Tuple[str, str]:
        """(output level, spatial key column) of a SWAT text output file name."""
        declared = self.TEXT_OUTPUT_FILES.get(filepath.name.lower())
        if declared is None:
            raise ValueError(
                f"{filepath.name} is not a SWAT text output table with an implemented layout; the "
                f"implemented layouts are {sorted(self.TEXT_OUTPUT_FILES)} (SWAT 2012 headout.f90, "
                "hrumon.f90, hruyr.f90 and hruaa.f90)."
            )
        return declared

    def _split_header_token(self, token: str) -> Tuple[str, Optional[str]]:
        """Column name and declared unit of one SWAT header field (for example ETmm, SOLARMJ/m2)."""
        name, units = token, None
        for suffix, declared_units in self.HEADER_UNIT_SUFFIXES:
            if token.endswith(suffix) and len(token) > len(suffix):
                name, units = token[: -len(suffix)], declared_units
                break
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_\-]*(?: [A-Za-z0-9_\-]+)*", name):
            raise ValueError(
                f"SWAT header field {token!r} is not a column name followed by a declared unit suffix "
                f"{[suffix for suffix, _ in self.HEADER_UNIT_SUFFIXES]}."
            )
        return name, units

    def _parse_swat_text_header(self, filepath: Path) -> Tuple[int, List[Tuple[str, Optional[str]]]]:
        """
        (header line index, [(column name, units)] of AREA and the variables) of output.hru.

        The header is headout.f90 format 1020: the identifier text up to AREAkm2, then one
        10-character field per variable. The ICALEN=1 header (format 1021) and any other line
        starting with LULC raise; a file without the header raises.
        """
        self._text_output_type(filepath)
        julian_layout = self.TEXT_HEADER_LAYOUTS[0]
        width = self.TEXT_FIELD_WIDTH
        with open(filepath, 'r', encoding="utf-8-sig") as handle:
            lines = handle.read().splitlines()
        for index, line in enumerate(lines):
            if line.startswith(self.TEXT_HEADER_LAYOUTS[1]):
                raise ValueError(
                    f"{filepath.name} line {index + 1} is the calendar-date header (headout.f90 format "
                    "1021, ICALEN=1); its row layout is not implemented (declare ICALEN=0)."
                )
            if not line.startswith(julian_layout):
                if line.startswith('LULC'):
                    raise ValueError(
                        f"{filepath.name} line {index + 1} is not the SWAT 2012 output.hru header layout "
                        f"{julian_layout!r} (headout.f90 format 1020)."
                    )
                continue
            fields = line[len(julian_layout):].rstrip()
            n_fields = -(-len(fields) // width)
            if n_fields == 0:
                raise ValueError(f"{filepath.name} header line {index + 1} declares no variable after AREAkm2.")
            fields = fields.ljust(n_fields * width)
            columns = [self._split_header_token('AREAkm2')]
            for k in range(n_fields):
                token = fields[k * width:(k + 1) * width].strip()
                if not token:
                    raise ValueError(f"{filepath.name} header line {index + 1}: variable field {k + 1} is blank.")
                columns.append(self._split_header_token(token))
            names = [name for name, _units in columns]
            if len(set(names)) != len(names):
                raise ValueError(f"{filepath.name} header line {index + 1} repeats a column name: {names}.")
            return index, columns
        raise ValueError(
            f"{filepath.name} has no SWAT 2012 output.hru header line ({julian_layout!r}, headout.f90 "
            "format 1020)."
        )

    @staticmethod
    def _fortran_number(field: str, where: str, label: str) -> float:
        """Value of one Fortran E or F output field; blank or overflowed (*) fields raise."""
        text = field.strip()
        if not text or '*' in text:
            raise ValueError(f"{where}: {label} {field!r} is blank or overflowed.")
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{where}: {label} {field!r} is not a number.") from exc

    def _parse_swat_text_row(self, line: str, n_values: int, where: str) -> Dict[str, Any]:
        """
        Fields of one output.hru row (hrumon/hruyr/hruaa formats 1000, 1001 and 2000).

        The HRU field is i5 (formats 1000 and 2000) or i7 (format 1001): the width is the one
        whose 1x separators (after HRU, SUB and MGT) are blank and whose HRU, SUB and MGT fields
        are integers. MON is kept as written (i4, or f4.1 on average-annual rows). A row fitting
        neither width, a short row, a blank, overflowed or non-numeric field and trailing text
        other than the (1x,i4) year of format 2000 raise.
        """
        width = self.TEXT_FIELD_WIDTH
        for hru_width in self.TEXT_ROW_HRU_WIDTHS:
            if (
                len(line) >= 39 + hru_width
                and line[4 + hru_width] == ' '
                and line[19 + hru_width] == ' '
                and line[24 + hru_width] == ' '
                and line[4:4 + hru_width].strip().isdigit()
                and line[14 + hru_width:19 + hru_width].strip().isdigit()
                and line[20 + hru_width:24 + hru_width].strip().isdigit()
            ):
                break
        else:
            raise ValueError(
                f"{where}: the row fits neither output.hru row format (a4,i5 or i7,1x,a5,a4,i5,1x,i4,1x,"
                "MON,e10.5,value fields)."
            )
        start = 39 + hru_width
        offsets = [
            start + width * k + sum(1 for spaced in self.TEXT_ROW_SPACED_VALUES if k >= spaced)
            for k in range(n_values)
        ]
        end = offsets[-1] + width
        if len(line) < end:
            raise ValueError(
                f"{where}: {len(line)} characters; the {n_values} value fields of the header end at "
                f"column {end}."
            )
        for spaced in self.TEXT_ROW_SPACED_VALUES:
            if spaced < n_values and line[offsets[spaced] - 1] != ' ':
                raise ValueError(f"{where}: column {offsets[spaced]} must be the 1x before value field {spaced + 1}.")
        tail = line[end:].rstrip()
        year = None
        if tail:
            if len(tail) != 5 or tail[0] != ' ' or not tail[1:].strip().isdigit():
                raise ValueError(
                    f"{where}: text {tail!r} follows the {n_values} value fields; only the (1x,i4) year "
                    "of format 2000 may follow them."
                )
            year = int(tail[1:])
        numbers = [self._fortran_number(line[29 + hru_width:39 + hru_width], where, 'AREA field')]
        numbers.extend(
            self._fortran_number(line[offset:offset + width], where, f"value field {k + 1}")
            for k, offset in enumerate(offsets)
        )
        return {
            'lulc': line[0:4].strip(),
            'hru': int(line[4:4 + hru_width]),
            'gis': line[5 + hru_width:14 + hru_width].strip(),
            'sub': int(line[14 + hru_width:19 + hru_width]),
            'mgt': int(line[20 + hru_width:24 + hru_width]),
            'mon': line[25 + hru_width:29 + hru_width],
            'numbers': numbers,
            'year': year,
        }

    def _check_file_cio_label(
        self, record: str, line_number: int, variable: str, source_line: int, path: Path
    ) -> None:
        """Raise when a writer label names a print setting other than the value SWAT reads there."""
        match = self.FILE_CIO_LABEL.search(record)
        if match is None:
            return
        label = match.group(1).upper()
        name = variable.upper()
        if label != name and (name in _SWAT_PRINT_SETTING_DOMAINS or label in _SWAT_PRINT_SETTING_DOMAINS):
            raise ValueError(
                f"{path.name} line {line_number}: the label {match.group(1)!r} disagrees with the value "
                f"SWAT reads from this record ({variable}, readfile.f90 line {source_line})."
            )

    def _file_cio_list_read(
        self, records: List[str], position: int, count: int, variable: str, source_line: int, path: Path
    ) -> Optional[Tuple[List[Optional[str]], int, int]]:
        """
        (values, line of the record holding the last value, next position) of one list-directed
        read of ``count`` values from ``records[position:]``; None at the end of the file.

        Blank records give no value, r*c gives r values (r* r null values), a slash ends the read
        and the rest of the last record is ignored; a value that is not a Fortran numeric constant
        raises (every list-directed variable of file.cio is numeric).
        """
        values: List[Optional[str]] = []
        line_number = position
        while len(values) < count:
            if position >= len(records):
                return None
            record = records[position]
            position += 1
            line_number = position
            terminated = False
            for token in re.split(r"[\s,]+", record.strip()):
                if not token:
                    continue
                if '/' in token:
                    token, terminated = token.split('/', 1)[0], True
                if token:
                    repeat = re.fullmatch(r"(\d+)\*(.*)", token)
                    constants = [repeat.group(2) or None] * int(repeat.group(1)) if repeat else [token]
                    for constant in constants:
                        if constant is not None and not self.FORTRAN_NUMBER.fullmatch(constant):
                            raise ValueError(
                                f"{path.name} line {line_number}: {constant!r} is not a numeric value for "
                                f"{variable} (list-directed read, readfile.f90 line {source_line}); the "
                                "file does not follow the readfile.f90 record sequence."
                            )
                    values.extend(constants)
                if terminated or len(values) >= count:
                    break
            if values or terminated:
                self._check_file_cio_label(record, line_number, variable, source_line, path)
            if terminated:
                break
        return values[:count], line_number, position

    def _read_file_cio(self, path: Path) -> Dict[str, Tuple[int, str]]:
        """
        Print settings of file.cio, read by replaying the readfile.f90 read sequence.

        SWAT reads file.cio by position (FILE_CIO_READS); the value of a setting is the first
        value of its record, whatever label the record carries. Returns {NAME: (value, source)}.
        A file ending before a read without iostat, a non-numeric list value, a non-integer
        setting, a setting outside its documented domain and a writer label contradicting the
        record position raise. ICALEN keeps its initial value 0 when the file ends before it.
        """
        with open(path, 'r', encoding="utf-8-sig") as handle:
            records = handle.read().splitlines()
        position = 0
        declared: Dict[str, Tuple[int, str]] = {}
        for source_line, edit, items, variable, iostat in self.FILE_CIO_READS:
            name = variable.upper()
            if edit == 'list':
                read = self._file_cio_list_read(records, position, items, variable, source_line, path)
                if read is None:
                    if iostat:
                        break
                    raise ValueError(
                        f"{path.name} reaches its end of file before the read of {variable} (readfile.f90 "
                        f"line {source_line}), which SWAT requires."
                    )
                values, line_number, position = read
                if name in _SWAT_PRINT_SETTING_DOMAINS:
                    token = values[0] if values else None
                    if token is None or not re.fullmatch(r"[+-]?\d+", token):
                        raise ValueError(
                            f"{path.name} line {line_number}: {name} (readfile.f90 line {source_line}) "
                            f"reads {token!r}, not an integer."
                        )
                    value = int(token)
                    _validate_swat_print_setting(name, value, f"{path.name} line {line_number}")
                    declared[name] = (value, f"{path.name} line {line_number}")
                continue
            n_records = -(-items // self.FILE_CIO_FORMAT_ITEMS[edit])
            if position + n_records > len(records):
                if iostat:
                    break
                raise ValueError(
                    f"{path.name} reaches its end of file before the read of {variable} (readfile.f90 "
                    f"line {source_line}), which SWAT requires."
                )
            for offset in range(n_records):
                self._check_file_cio_label(records[position + offset], position + offset + 1, variable, source_line, path)
            position += n_records
        for name, value in self.FILE_CIO_INITIAL_VALUES.items():
            if name not in declared:
                declared[name] = (
                    value,
                    f"{path.name} ends before its read (iostat=eof), initial value from zero0.f90",
                )
        return declared

    def _swat_print_settings(self) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Declared print settings of the SWAT run and where each one was declared.

        root_path/file.cio (read as SWAT reads it, _read_file_cio) is used where present;
        SWATConfig declares the settings otherwise, and the calendar. A configured value different from file.cio, an undeclared setting and
        an undeclared calendar raise; nothing is inferred.
        """
        file_cio = Path(self.config.root_path) / 'file.cio'
        in_file = self._read_file_cio(file_cio) if file_cio.is_file() else {}
        settings: Dict[str, Any] = {}
        sources: Dict[str, str] = {}
        missing: List[str] = []
        for name in _SWAT_PRINT_SETTING_DOMAINS:
            configured = getattr(self.config, name.lower())
            declared = in_file.get(name)
            if declared is not None and configured is not None and declared[0] != configured:
                raise ValueError(
                    f"SWAT {name}: {file_cio} declares {declared[0]} ({declared[1]}), "
                    f"SWATConfig.{name.lower()} declares {configured}."
                )
            if declared is not None:
                settings[name], sources[name] = declared
            elif configured is not None:
                settings[name], sources[name] = configured, 'configuration'
            else:
                missing.append(name)
        if missing:
            raise ValueError(
                f"SWAT text output needs the print settings {missing} declared in {file_cio} or in "
                f"SWATConfig ({[name.lower() for name in missing]}); none is inferred."
            )
        if self.config.calendar is None:
            raise ValueError(
                "SWAT text output needs SWATConfig.calendar, the CF calendar of the run (file.cio "
                "declares none); none is inferred."
            )
        settings['calendar'], sources['calendar'] = self.config.calendar, 'configuration'
        # References: SWAT 2012 rev 670 readfile.f90 line 293.
        # Mathematical expression: NYSKIP' = NBYR - 1 if NYSKIP >= NBYR, else NYSKIP' = NYSKIP.
        # Parameter meanings: NYSKIP warm-up years not printed; NBYR simulated years; NYSKIP'
        #   the value SWAT runs with.
        # Units: years.
        # Domain/constraints: NBYR >= 1, so NYSKIP' >= 0 and at least the last year is printed.
        # Sign convention: not applicable.
        if settings['NYSKIP'] >= settings['NBYR']:
            settings['NYSKIP'] = settings['NBYR'] - 1
            sources['NYSKIP'] += " (NYSKIP >= NBYR: set to NBYR - 1 as SWAT does, readfile.f90 line 293)"
        return settings, sources

    def _swat_print_sequence(
        self, settings: Dict[str, Any], filepath: Path
    ) -> Tuple[List[Dict[str, Any]], str, str]:
        """
        (expected output.hru blocks, time units, calendar) of the declared SWAT run.

        SWAT 2012 rev 670 prints only the simulated years curyr > NYSKIP (simulate.f90): the
        daily blocks with IPRINT=1 (hruday, virtual.f90; MON = julian day), the monthly blocks
        with IPRINT=0 (hrumon, writem.f90; MON = month), with IPRINT /= 1 one annual block per
        year (hruyr, writea.f90; MON = year) and, with IPRINT /= 1, one average-annual block at
        the end (hruaa, writeaa.f90; MON = years averaged). A year is simulated from day IDAF
        (first year, IDAF > 0) or 1 to day IDAL (last year, IDAL > 0) or its last day
        (simulate.f90). Every block carries its kind, MON, year, CF time bounds and time value;
        the period boundaries come from PeriodIndexer under the declared calendar, which must
        give SWAT's year and month lengths; otherwise it raises.
        """
        nbyr, iyr, idaf, idal, iprint, nyskip = (
            settings[key] for key in ('NBYR', 'IYR', 'IDAF', 'IDAL', 'IPRINT', 'NYSKIP')
        )
        last_year = iyr + nbyr - 1
        if last_year > _SWAT_PRINT_SETTING_DOMAINS['IYR'][1]:
            raise ValueError(
                f"SWAT IYR={iyr} and NBYR={nbyr} end in {last_year}, beyond the four-digit years of "
                "output.hru."
            )
        time_units = f"days since {iyr:04d}-01-01 00:00:00"
        indexer = self.time_handler.period_indexer
        cf_calendar = settings['calendar']
        expected: List[Dict[str, Any]] = []
        printed_years = 0
        for cycle, year in enumerate(range(iyr, last_year + 1), start=1):
            period = (f"{year:04d}-01-01", f"{year + 1:04d}-01-01", "M")
            declared = indexer.create_period_index(*period, settings['calendar'], time_units)
            swat = indexer.create_period_index(*period, self.SWAT_CALENDAR, time_units)
            cf_calendar = declared.calendar
            months = declared.bounds.detach().cpu().numpy().astype(np.float64)
            month_days = np.diff(months, axis=1).ravel()
            swat_days = np.diff(swat.bounds.detach().cpu().numpy().astype(np.float64), axis=1).ravel()
            if month_days.shape != swat_days.shape or not np.array_equal(month_days, swat_days):
                raise ValueError(
                    f"The declared calendar {settings['calendar']!r} gives {year} the month lengths "
                    f"{month_days.tolist()} days; SWAT 2012 simulates {swat_days.tolist()} (leap year when "
                    "mod(year, 4) = 0, simulate.f90)."
                )
            year_days = int(round(months[-1, 1] - months[0, 0]))
            first_day = idaf if cycle == 1 and idaf > 0 else 1
            last_day = idal if cycle == nbyr and idal > 0 else year_days
            if first_day > year_days or last_day > year_days or last_day < first_day:
                raise ValueError(
                    f"SWAT simulates {year} from day {first_day} to day {last_day} (IDAF={idaf}, "
                    f"IDAL={idal}); the year has {year_days} days."
                )
            if cycle <= nyskip:
                continue

            # References: SWAT 2012 rev 670 simulate.f90 (id1, idlst, curyr > nyskip), virtual.f90
            #   (hruday), writem.f90 (hrumon), writea.f90 (hruyr), writeaa.f90 (hruaa); CF
            #   Conventions 1.8 sections 4.4 (time coordinate, calendar) and 7.1 (cell boundaries);
            #   PeriodIndexer.create_period_index, the calendar owner of the month boundaries.
            # Mathematical expression: W_y = [t(y) + d1 - 1, t(y) + d2) for a printed year y; a
            #   monthly record of month P = [a, b) has the bounds [max(a, W_y0), min(b, W_y1)) and
            #   exists iff min(b, W_y1) > max(a, W_y0); a daily record of day d has
            #   [t(y) + d - 1, t(y) + d); an annual record has W_y; the time value of a record is
            #   the cell centre (lower + upper) / 2.
            # Parameter meanings: t(y) start of year y; d1 = IDAF in the first simulated year when
            #   IDAF > 0, else 1; d2 = IDAL in the last simulated year when IDAL > 0, else the
            #   number of days of y; y is printed iff its cycle curyr > NYSKIP.
            # Units: all times in days since IYR-01-01 00:00:00 of the declared CF calendar.
            # Domain/constraints: 1 <= d1 <= d2 <= days of y; the declared calendar gives SWAT's
            #   month lengths for every simulated year.
            # Sign convention: bounds increase with time (lower < upper).
            year_start = months[0, 0]
            window = (year_start + first_day - 1, year_start + last_day)
            periods: List[Tuple[int, float, float]] = []
            if iprint == 0:
                for index in range(months.shape[0]):
                    lower, upper = max(months[index, 0], window[0]), min(months[index, 1], window[1])
                    if upper > lower:
                        periods.append((index + 1, lower, upper))
            elif iprint == 1:
                periods = [(day, year_start + day - 1, year_start + day) for day in range(first_day, last_day + 1)]
            for mon, lower, upper in periods:
                expected.append({
                    'kind': 'period', 'mon': mon, 'year': year,
                    'bounds': (lower, upper), 'value': 0.5 * (lower + upper),
                })
            if iprint != 1:
                expected.append({
                    'kind': 'annual', 'mon': year, 'year': year,
                    'bounds': window, 'value': 0.5 * (window[0] + window[1]),
                })
            printed_years += 1
        if iprint != 1:
            expected.append({'kind': 'average', 'mon': None, 'year': last_year, 'years': printed_years})
        return expected, time_units, cf_calendar

    def _describe_swat_block(self, block: Dict[str, Any], iprint: int) -> str:
        """Text naming one expected output.hru block."""
        if block['kind'] == 'period':
            return f"the {self.PRINT_FREQUENCIES[iprint]} block MON={block['mon']} of {block['year']}"
        if block['kind'] == 'annual':
            return f"the annual block MON={block['year']}"
        return f"the average-annual block (MON = years averaged as f4.1, at most {block['years']})"

    def _classify_swat_mon(self, mon: str, iprint: int) -> str:
        """Which rule of IPRINT a MON field fits, or that it fits none."""
        text = mon.strip()
        if iprint != 1 and re.fullmatch(r"\d{1,2}\.\d", text):
            return "the years averaged of an average-annual block"
        if text.isdigit():
            value = int(text)
            if iprint != 1 and len(text) == 4 and value >= _SWAT_PRINT_SETTING_DOMAINS['IYR'][0]:
                return "the four-digit year of an annual block"
            if 1 <= value <= self.PERIOD_MON_LIMITS[iprint]:
                return f"a {'month' if iprint == 0 else 'julian day'}"
        return f"it fits no rule of IPRINT={iprint}: MON is {self.MON_RULES[iprint]}"

    def _match_swat_blocks(
        self,
        blocks: List[Dict[str, Any]],
        expected: List[Dict[str, Any]],
        settings: Dict[str, Any],
        filepath: Path,
    ) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any]]], Optional[float]]:
        """
        ((file block, expected block) of every record, years averaged) of an output.hru file.

        The file's blocks must be the declared print sequence one by one (_swat_print_sequence):
        the MON of a monthly, daily or annual block equals the expected month, day or year, the
        last block of IPRINT 0 and 2 is the average-annual block (identified by its position;
        MON = years averaged as f4.1, at most the printed years) and the (1x,i4) year of format
        2000 rows equals the block year. Any deviation, a missing block and an extra block raise.
        The records are the monthly or daily blocks (IPRINT 0, 1) or the annual blocks (IPRINT 2).
        """
        name = filepath.name
        iprint = settings['IPRINT']
        declared = ", ".join(f"{key}={settings[key]}" for key in ('IPRINT', 'NBYR', 'IYR', 'IDAF', 'IDAL', 'NYSKIP'))
        years_averaged: Optional[float] = None
        for index, (block, want) in enumerate(zip(blocks, expected)):
            text = block['mon'].strip()
            if want['kind'] == 'average':
                matched = bool(re.fullmatch(r"\d{1,2}\.\d", text)) and float(text) <= want['years']
                if matched:
                    years_averaged = float(text)
            else:
                matched = text.isdigit() and int(text) == want['mon']
            if not matched:
                raise ValueError(
                    f"{name} line {block['line']}: block {index + 1} has MON {block['mon']!r} "
                    f"({self._classify_swat_mon(block['mon'], iprint)}); the declared print sequence "
                    f"({declared}) expects {self._describe_swat_block(want, iprint)} there."
                )
            if want['kind'] != 'average':
                for row in block['rows']:
                    if row['year'] is not None and row['year'] != want['year']:
                        raise ValueError(
                            f"{name} line {block['line']}: a row of block {index + 1} appends the year "
                            f"{row['year']}; the block belongs to {want['year']}."
                        )
        if len(blocks) > len(expected):
            extra = blocks[len(expected)]
            raise ValueError(
                f"{name} line {extra['line']}: block {len(expected) + 1} (MON {extra['mon']!r}) follows the "
                f"end of the declared print sequence ({declared}), which has {len(expected)} blocks."
            )
        if len(blocks) < len(expected):
            raise ValueError(
                f"{name} ends after {len(blocks)} blocks; the declared print sequence ({declared}) has "
                f"{len(expected)}, the next being {self._describe_swat_block(expected[len(blocks)], iprint)}."
            )
        record_kind = 'annual' if iprint == 2 else 'period'
        records = [(block, want) for block, want in zip(blocks, expected) if want['kind'] == record_kind]
        return records, years_averaged

    def _read_swat_text_file(self, filepath: Path) -> xr.Dataset:
        """
        Read one SWAT 2012 output.hru table.

        Fixed-width parsing per the header and row formats above; blocks are runs of rows with
        one MON field and list the same HRUs; the blocks must be the print sequence of the
        declared settings (_swat_print_settings, _swat_print_sequence, _match_swat_blocks). The
        time series holds the monthly or daily blocks (IPRINT=0 or 1) or the annual blocks
        (IPRINT=2) with CF time bounds; the annual and average-annual summary blocks are
        excluded and counted in the attributes. Malformed input raises.
        """
        _output_type, key = self._text_output_type(filepath)
        settings, sources = self._swat_print_settings()
        if settings['ICALEN'] == 1:
            raise ValueError(
                "SWAT ICALEN=1 (month/day/year rows under the headout.f90 format 1021 header, which "
                "also sets IPRINT=1, readfile.f90 line 741) is not implemented: the verified row "
                "formats carry MON; declare ICALEN=0."
            )
        if self.config.record_interval is not None:
            raise ValueError(
                "SWAT text output record bounds follow IPRINT, the declared simulation window and the "
                f"calendar; the configured record_interval ({self.config.record_interval} "
                f"{self.config.record_interval_units}) is a second, fixed declaration and is rejected."
            )
        iprint = settings['IPRINT']
        header_index, columns = self._parse_swat_text_header(filepath)
        with open(filepath, 'r', encoding="utf-8-sig") as handle:
            lines = handle.read().splitlines()

        blocks: List[Dict[str, Any]] = []
        for line_number, line in enumerate(lines[header_index + 1:], start=header_index + 2):
            if not line.strip():
                continue
            row = self._parse_swat_text_row(line, len(columns) - 1, f"{filepath.name} line {line_number}")
            if not blocks or blocks[-1]['mon'] != row['mon']:
                blocks.append({'mon': row['mon'], 'line': line_number, 'rows': []})
            blocks[-1]['rows'].append(row)
        if not blocks:
            raise ValueError(f"{filepath.name} has no data rows after its header.")
        units_listed = [(row['hru'], row['gis'], row['sub'], row['mgt']) for row in blocks[0]['rows']]
        for block in blocks:
            listed = [(row['hru'], row['gis'], row['sub'], row['mgt']) for row in block['rows']]
            hrus = [entry[0] for entry in listed]
            if len(set(hrus)) != len(hrus):
                raise ValueError(
                    f"{filepath.name} line {block['line']}: the block MON {block['mon']!r} lists an HRU "
                    f"more than once: {hrus}."
                )
            if listed != units_listed:
                raise ValueError(
                    f"{filepath.name} line {block['line']}: the block MON {block['mon']!r} lists the HRUs "
                    f"{hrus}, the first block {[entry[0] for entry in units_listed]} (with GIS, SUB, MGT)."
                )
        expected, time_units, cf_calendar = self._swat_print_sequence(settings, filepath)
        records, years_averaged = self._match_swat_blocks(blocks, expected, settings, filepath)
        bounds = np.asarray([want['bounds'] for _block, want in records], dtype=np.float64)
        time_values = np.asarray([want['value'] for _block, want in records], dtype=np.float64)

        spatial_dim = key.lower()
        data_vars = {}
        for position, (name, units) in enumerate(columns):
            values = np.array(
                [[row['numbers'][position] for row in block['rows']] for block, _want in records], dtype=np.float64
            )
            data_vars[name] = (('time', spatial_dim), values, {'units': units} if units is not None else {})
        coords = {
            'time': ('time', time_values, {'units': time_units, 'calendar': cf_calendar, 'bounds': 'time_bnds'}),
            'time_bnds': (('time', 'nv'), bounds),
            self.TEXT_RECORD_COLUMN: ('time', np.array([want['mon'] for _block, want in records], dtype=np.int64)),
            'YR': ('time', np.array([want['year'] for _block, want in records], dtype=np.int64)),
            spatial_dim: (spatial_dim, np.array([entry[0] for entry in units_listed], dtype=np.int64)),
            'GIS': (spatial_dim, np.array([entry[1] for entry in units_listed])),
            'SUB': (spatial_dim, np.array([entry[2] for entry in units_listed], dtype=np.int64)),
            'MGT': (spatial_dim, np.array([entry[3] for entry in units_listed], dtype=np.int64)),
            # The land cover can change between records (crop rotation), so LULC is per record.
            'LULC': (('time', spatial_dim), np.array([[row['lulc'] for row in block['rows']] for block, _want in records])),
        }
        dataset = xr.Dataset(data_vars, coords=coords)
        dataset.attrs.update({
            'swat_output_file': filepath.name,
            'swat_print_frequency': self.PRINT_FREQUENCIES[iprint],
            'swat_iprint': iprint,
            'swat_icalen': settings['ICALEN'],
            'swat_nbyr': settings['NBYR'],
            'swat_iyr': settings['IYR'],
            'swat_idaf': settings['IDAF'],
            'swat_idal': settings['IDAL'],
            'swat_nyskip': settings['NYSKIP'],
            'swat_print_settings_source': ', '.join(f"{name}={source}" for name, source in sources.items()),
            'swat_record_blocks': len(records),
            'swat_annual_summary_blocks': sum(1 for want in expected if want['kind'] == 'annual') if iprint == 0 else 0,
            'swat_average_annual_summary_blocks': sum(1 for want in expected if want['kind'] == 'average'),
        })
        if years_averaged is not None:
            dataset.attrs['swat_average_annual_years'] = years_averaged
        return dataset

    def _apply_subsetting(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply temporal and spatial subsetting"""
        
        # Apply temporal subsetting (numeric CF time records are not selected by a date window)
        if self.time_window and 'time' in ds.coords:
            if ds['time'].dtype.kind != 'M':
                raise ValueError(
                    "SWAT records carry a numeric CF time coordinate; selecting them by a date window is "
                    "not implemented."
                )
            ds = ds.sel(time=slice(self.time_window.start, self.time_window.end))
        
        # Spatial subsetting for SWAT is based on subbasin/HRU IDs
        # This would require additional watershed information
        
        return ds
    
    def _add_swat_metadata(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Complete SWAT metadata without overriding the file.

        Attributes the file declares are authoritative; the SWAT table supplies long names, and
        units only where the file declares none (recorded as ``units_source = adapter_table``;
        SWAT text and SQLite outputs carry no units attributes).
        """
        filled = self._apply_adapter_attributes(ds, self.variable_definitions, "SWAT hydrological model")
        if filled:
            self.logger.info("SWAT units taken from the adapter table (none in the file): %s", filled)

        if 'time' in ds.coords:
            for key, value in {'standard_name': 'time', 'long_name': 'Time'}.items():
                ds['time'].attrs.setdefault(key, value)

        for key, value in {
            'source': 'SWAT hydrological model',
            'model_version': 'SWAT',
            'output_level': self.config.output_level,
        }.items():
            ds.attrs.setdefault(key, value)
        return ds

    def _get_temporal_coverage(self) -> Dict[str, Dict]:
        """Get temporal coverage for all datasets"""
        coverage = {}
        
        for dataset_name, ds in self.current_datasets.items():
            if 'time' in ds.coords:
                time_coord = ds.coords['time']
                coverage[dataset_name] = {
                    "start": str(time_coord.values[0]),
                    "end": str(time_coord.values[-1]),
                    "n_timesteps": len(time_coord)
                }
        
        return coverage
    
    def _get_all_variables(self) -> Dict[str, List[str]]:
        """Get all available variables across datasets"""
        variables = {}
        
        for dataset_name, ds in self.current_datasets.items():
            variables[dataset_name] = list(ds.data_vars.keys())
        
        return variables
    
    def read(
        self, 
        variables: Optional[List[str]] = None,
        dataset: Optional[str] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Read and transform SWAT data to RELIC internal format.
        
        Parameters:
            variables: List of variables to read (internal names)
            dataset: Specific dataset to read from
            **kwargs: Additional reading parameters
            
        Returns:
            Dictionary mapping internal variable names to PyTorch tensors
        """
        if not self.current_datasets:
            raise ValueError("No datasets opened. Call open() first.")
        
        # Select dataset(s) to read from
        if dataset:
            datasets_to_read = {dataset: self.current_datasets[dataset]}
        else:
            datasets_to_read = self.current_datasets
        
        self.logger.info(f"Reading SWAT variables from {len(datasets_to_read)} datasets")
        
        # Get variable mappings
        from ..common.schema import InternalSchema
        # internal -> SWAT through the schema owner (ambiguous many-to-one mappings raise
        # unless a canonical external variable is declared).
        internal_to_swat = InternalSchema().resolve_external_names(DataSource.SWAT, variables)
        
        tensors = {}
        
        # Mapped fields each dataset contains; a requested field that no dataset has raises
        # and absent mapped fields are listed in last_read_report (never skipped silently).
        present = self._present_mapped_fields(datasets_to_read, internal_to_swat, variables)
        for dataset_name, ds in datasets_to_read.items():
            dataset_tensors = {}
            
            for internal_var in present[dataset_name]:
                swat_var = internal_to_swat[internal_var]
                
                # Declared source transform (if any) and unit conversion: the one read path.
                converted_data = self._read_field(ds, swat_var, internal_var)

                # Convert to PyTorch tensor
                tensor = torch.from_numpy(converted_data.values.astype(np.float32))

                # Add metadata
                tensor_meta = {
                    'shape': tensor.shape,
                    'dims': converted_data.dims,
                    'coords': {
                        name: torch.from_numpy(coord.values.astype(np.float32))
                        for name, coord in converted_data.coords.items()
                        if coord.dtype.kind in 'fc'
                    },
                    'coord_dims': {
                        name: tuple(coord.dims)
                        for name, coord in converted_data.coords.items()
                        if coord.dtype.kind in 'fc'
                    },
                    'variable': internal_var,
                    **self._record_time_metadata(converted_data),
                    'cell_bounds': self._cf_horizontal_coordinates(ds, converted_data),
                    'attrs': dict(converted_data.attrs),
                    'dataset': dataset_name
                }

                setattr(tensor, '_relic_meta', tensor_meta)
                dataset_tensors[internal_var] = tensor
                
            
            # Merge dataset tensors
            if len(datasets_to_read) == 1:
                tensors.update(dataset_tensors)
            else:
                for var_name, tensor in dataset_tensors.items():
                    key = f"{var_name}_{dataset_name}"
                    tensors[key] = tensor
        
        return tensors
    
    def transform_to_internal(
        self,
        data: Dict[str, torch.Tensor],
        target_grid: Optional[Dict] = None,
        variable_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Transform SWAT data to RELIC internal format (shared ExternalAdapter._transform_fields)."""
        if kwargs:
            raise TypeError(f"Unsupported transform options: {sorted(kwargs)}")
        return self._transform_fields(data, target_grid=target_grid, variable_metadata=variable_metadata)

    def close(self) -> None:
        """Close all opened datasets"""
        self.logger.info("Closing SWAT adapter")
        
        # Every dataset is attempted; the state is cleared and close errors are raised.
        self._close_datasets(self.discovered_files, self.watershed_info)

    def _select_main_dataset(self) -> Tuple[Optional[str], Optional[xr.Dataset]]:
        """The dataset of the configured output level when it carries a schema-mapped SWAT column,
        else (None, None)."""
        key = self.config.output_level
        dataset = (self.current_datasets or {}).get(key)
        if dataset is not None and self._carries_mapped_columns(dataset):
            return key, dataset
        return None, None

    def _carries_mapped_columns(self, dataset: xr.Dataset) -> bool:
        """Return whether a dataset has at least one column of the SWAT schema mapping."""
        from ..common.schema import InternalSchema

        data_vars = getattr(dataset, 'data_vars', {})
        return any(column in data_vars for column in InternalSchema().get_source_mapping(DataSource.SWAT))

    def forward(
        self,
        state: Dict[str, torch.Tensor],
        time: float,
        dt: float,
        *,
        calendar_type: Optional[str] = None,
        time_origin: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for SWAT data provision at the simulation time.

        Returns every catalog field of the opened SWAT data at time_origin + ``time`` in the
        simulator calendar through the single adapter read path (schema mapping, declared source
        transforms, unit conversion, then the configured time_interpolation);
        ``kwargs["variables"]`` restricts the internal variable names. The simulator passes
        ``calendar_type`` and ``time_origin`` from its settings; both are required. The dataset is the one of the configured output level (_select_main_dataset); without opened
        datasets, or without one of the output level carrying a schema-mapped SWAT column, the
        provided variables cannot be read and a ValueError is raised (no empty result).

        Returns:
            Dict[str, UpdateItem]: replace updates keyed by internal catalog variable names
        """
        # References: RELIC coupling adapter contract (relic/coupling/external/common/base_adapter.py,
        #   ExternalAdapter.read, _read_field and _read_records_at); CDEPS/E3SM shr_tInterp_getFactors
        #   (forcing_record_weights); no SWAT process equation is evaluated here.
        # Mathematical expression: y_n(t) = sum_(k, w) w X_n[record k] with X_n = read(variables)[n]
        #   (X_n = convert(transform(D[m(n)])), m the schema mapping) and
        #   (k, w) = forcing_record_weights(s, t, rule), s the record times in seconds since time_origin.
        # Parameter meanings: D opened SWAT dataset; m(n) external name of internal variable n;
        #   transform the declared source transform; convert the catalog unit conversion; rule the
        #   configured time_interpolation ('lower' or 'linear').
        # Units: catalog units of every internal variable (converted by the read path); time [s].
        # Domain/constraints: undeclared fields, missing units and unconvertible units raise in the
        #   read path; a missing time_origin, calendar_type or time_interpolation, a dataset calendar
        #   other than the simulator calendar and a time outside the records raise; no read
        #   position is kept.
        # Sign convention: not applicable: external data provision, no flux direction is introduced.
        if not self.current_datasets:
            raise ValueError("No SWAT datasets are opened; call open() before forward.")
        main_key, main_ds = self._select_main_dataset()
        if main_ds is None:
            raise ValueError(
                f"No opened SWAT dataset of the output level {self.config.output_level!r} carries mapped "
                f"SWAT columns (keys={sorted(self.current_datasets)}); the provided variables "
                f"{list(self.provided_variables)} cannot be read."
            )
        self.logger.debug("SWAT forward uses dataset '%s'", main_key)
        return self._read_records_at(
            time,
            calendar_type=calendar_type,
            time_origin=time_origin,
            variables=kwargs.get("variables"),
            dataset=main_key,
        )


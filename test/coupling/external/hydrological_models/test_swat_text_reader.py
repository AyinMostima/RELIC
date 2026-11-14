"""Owner tests for the SWAT 2012 output.hru reader (fixed-width layout and block sequence).

Layout (SWAT 2012 rev 670 source): headout.f90 format 1020 writes the header
'LULC  HRU       GIS  SUB  MGT  MON' + '   AREAkm2' followed by one 10-character ``heds``
field per variable; hrumon.f90 and hruyr.f90 format 1000 write the rows
(a4,i5,1x,a5,a4,i5,1x,i4,1x,i4,e10.5,66f10.3,1x,e10.5,1x,e10.5,...), hruyr.f90 format 1001
writes HRU as i7 and hruaa.f90 writes MON as f4.1. MON is the julian day (IPRINT=1), the
month (IPRINT=0), the four-digit year (annual summary lines, hruyr) or the number of years
averaged (average-annual summary lines, hruaa). The block sequence follows the SWAT source:
hruday only with IPRINT=1 (virtual.f90), hrumon only with IPRINT=0 (writem.f90), hruyr and
hruaa only with IPRINT /= 1 (writea.f90, writeaa.f90), all only for curyr > NYSKIP
(simulate.f90). The average-annual block is identified by its position (its MON can equal a
month). IDAF applies to the first simulated year only and IDAL to the last one; 0 means the
first or last day of the year (simulate.f90).

Record bounds (IPRINT=0: calendar months; IPRINT=1: days; IPRINT=2: years) follow from the
declared settings (NBYR, IYR, IDAF, IDAL, NYSKIP) and calendar through PeriodIndexer.

Reference: SWAT 2012 Input/Output Documentation (Arnold et al., 2012), section 32.3.
"""

from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pytest
import torch

from relic.coupling.external.hydrological_models.swat_adapter import SWATAdapter, SWATConfig

_IDENTIFIERS = "LULC  HRU       GIS  SUB  MGT  MON   AREAkm2"
_HEDS = ("  PRECIPmm", "      ETmm", "SURQ_GENmm", " LATQGENmm", " TMP_MXdgC", "SOL_TMPdgC", "SOLARMJ/m2", "       LAI")
_HEADER = _IDENTIFIERS + "".join(_HEDS)
# (LULC, HRU, GIS = subnum (a5) + hruno (a4), SUB, MGT, AREA written as Fortran e10.5)
_HRUS = (
    ("AGRL", 1, "00001", "0001", 1, 1, ".10700E+02"),
    ("FRSD", 2, "00001", "0002", 1, 1, ".52000E+01"),
)
_DEPTH_PER_DAY = (1.0, 2.0)  # mm of water per simulated day for HRU 1 and HRU 2
_SECONDS_PER_DAY = 86400.0
_MONTHLY = dict(iprint=0, icalen=0, nbyr=2, iyr=2021, idaf=1, idal=365, nyskip=0, calendar="standard")


def _file_cio_lines(
    nbyr=2, iyr=2021, idaf=1, idal=365, iprint=0, nyskip=0, icalen=0, labels=True
) -> list[str]:
    """file.cio in the SWAT 2012 layout read by readfile.f90 (85 records up to ICALEN).

    Value records carry the writer's 'value | NAME : description' label when ``labels``;
    SWAT itself reads the records by position and takes the first value of each.
    """

    def value(number, label: str) -> str:
        return f"{number:>16}    | {label}" if labels else f"{number:>16}"

    climate = (
        ("IGEN", 0), ("PCPSIM", 1), ("IDT", 0), ("IDIST", 0), ("REXP", "1.300"), ("NRGAGE", 1),
        ("NRTOT", 1), ("NRGFIL", 1), ("TMPSIM", 1), ("NTGAGE", 1), ("NTTOT", 1), ("NTGFIL", 1),
        ("SLRSIM", 2), ("NSTOT", 0), ("RHSIM", 2), ("NHTOT", 0), ("WNDSIM", 2), ("NWTOT", 0),
        ("FCSTYR", 0), ("FCSTDAY", 0), ("FCSTCYCLES", 0),
    )
    return [
        "Master Watershed File: file.cio",  # 1 titldum (readfile.f90 151)
        "Project Description:",  # 2 titldum (152)
        "General Input/Output section (file.cio):",  # 3-5 title, 20a4 x 60 (153)
        "9/23/2026 12:00:00 AM synthetic RELIC test run",
        "",
        "General Information/Watershed Configuration:",  # 6 titldum (156)
        "fig.fig",  # 7 figfile (157)
        value(nbyr, "NBYR : Number of years simulated"),  # 8 (158)
        value(iyr, "IYR : Beginning year of simulation"),  # 9 (159)
        value(idaf, "IDAF : Beginning julian day of simulation"),  # 10 (160)
        value(idal, "IDAL : Ending julian day of simulation"),  # 11 (161)
        "Climate:",  # 12 titldum (167)
        *[value(number, f"{name} : climate input") for name, number in climate],  # 13-33 (168-188)
        "Precipitation Files:",  # 34 titldum (189)
        "pcp1.pcp", "", "",  # 35-37 rfile, 6a x 18 (190)
        "Temperature Files:",  # 38 titldum (191)
        "tmp1.tmp", "", "",  # 39-41 tfile, 6a x 18 (192)
        "", "", "", "",  # 42-45 slrfile, rhfile, wndfile, fcstfile (193-196)
        "Watershed Modeling Options:",  # 46 titldum (256)
        "basins.bsn",  # 47 bsnfile (257)
        "Database Files:",  # 48 titldum (263)
        "plant.dat", "till.dat", "pest.dat", "fert.dat", "urban.dat",  # 49-53 (264-268)
        "Special Projects:",  # 54 titldum (283)
        value(0, "ISPROJ : special project"), value(0, "ICLB : auto-calibration option"), "",  # 55-57 (284-286)
        "Output Information:",  # 58 titldum (289)
        value(iprint, "IPRINT: print code (month, day, year)"),  # 59 (290)
        value(nyskip, "NYSKIP: number of years to skip output printing/summarization"),  # 60 (291)
        value(0, "ILOG: streamflow print code"), value(1, "IPRP: print code for output.pst file"),  # 61-62
        "",  # 63 titldum (297)
        "Reach output variables:",  # 64 titldum (307)
        "   0" * 20,  # 65 ipdvar (308)
        "Subbasin output variables:",  # 66 titldum (363)
        "   0" * 15,  # 67 ipdvab (364)
        "HRU output variables:",  # 68 titldum (385)
        "   0" * 20,  # 69 ipdvas (386)
        "HRU data to be printed:",  # 70 titldum (469)
        "   0" * 20,  # 71 ipdhru (470)
        "ATMOSPHERIC DEPOSITION",  # 72 titldum (474)
        "",  # 73 atmofile (476)
        *[value(0, name) for name in ("IPHR", "ISTO", "ISOL", "I_SUBHW")],  # 74-77 (485-508)
        "septwq.dat",  # 78 septdb (513)
        *[value(0, name) for name in ("IA_B", "IHUMUS", "ITEMP", "ISNOW", "IMGT", "IWTR")],  # 79-84 (526-723)
        value(icalen, "ICALEN: Code for printing out calendar or julian dates to .rch, .sub and .hru files"),  # 85 (740)
    ]


def _write_file_cio(root: Path, lines: list[str]) -> None:
    (root / "file.cio").write_text("\n".join(lines) + "\n", encoding="ascii")


def _row(hru_index: int, mon_field: str, precip: float, hru_width: int = 5, year: int | None = None) -> str:
    lulc, hru, subnum, hruno, sub, mgt, area = _HRUS[hru_index]
    values = (precip, precip / 4.0, precip / 5.0, precip / 10.0, 20.0 + hru_index, 5.0 + hru_index, 12.5, 1.2 + 0.1 * hru_index)
    line = f"{lulc:<4}{hru:>{hru_width}d} {subnum:>5}{hruno:>4}{sub:>5d} {mgt:>4d} {mon_field}{area}"
    line += "".join(f"{value:10.3f}" for value in values)
    return line + (f" {year:4d}" if year is not None else "")


def _block(mon_field: str, days: float, hru_width: int = 5, year: int | None = None) -> list[str]:
    """One record block: every HRU with the water depth of ``days`` simulated days."""
    assert len(mon_field) == 4
    return [_row(index, mon_field, days * rate, hru_width, year) for index, rate in enumerate(_DEPTH_PER_DAY)]


def _monthly_blocks(years=(2021, 2022), first_day: int = 1) -> list[list[str]]:
    """IPRINT=0: 12 month blocks and one annual block per year, then the average-annual block.

    The average-annual MON is 2.0 years, the value of February.
    """
    blocks = []
    for year in years:
        for month in range(1, 13):
            days = calendar.monthrange(year, month)[1]
            if year == years[0] and month == 1:
                days -= first_day - 1
            blocks.append(_block(f"{month:4d}", days))
        blocks.append(_block(f"{year:4d}", 365.0))
    blocks.append(_block(f"{float(len(years)):4.1f}", 365.0))
    return blocks


def _write_output_hru(root: Path, blocks, header: str = _HEADER) -> Path:
    lines = [
        "1",
        "    SWAT Sep 7    VER 2018/Rev 670",
        "",
        "    synthetic RELIC test output",
        "",
        "",
        header,
    ]
    for block in blocks:
        lines.extend(block)
    path = root / "output.hru"
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def _open(root: Path, **config) -> SWATAdapter:
    adapter = SWATAdapter(SWATConfig(root_path=root, output_level="hru", **config))
    adapter.discovered_files = [root / "output.hru"]
    adapter.open()
    return adapter


def _flux_per_hru(n_records: int) -> torch.Tensor:
    """F = rho_w d / dt for one depth per day: 1 and 2 mm day-1 -> 1/86400 and 2/86400 kg m-2 s-1."""
    return torch.tensor([list(_DEPTH_PER_DAY)] * n_records, dtype=torch.float32) / _SECONDS_PER_DAY


def test_monthly_output_hru_excludes_annual_and_average_annual_blocks(tmp_path: Path) -> None:
    _write_output_hru(tmp_path, _monthly_blocks())
    dataset = _open(tmp_path, **_MONTHLY).current_datasets["hru"]
    assert dataset.sizes["time"] == 24
    assert dataset["PRECIP"].dims == ("time", "hru")
    np.testing.assert_array_equal(dataset["MON"].values, list(range(1, 13)) * 2)
    np.testing.assert_array_equal(dataset["YR"].values, [2021] * 12 + [2022] * 12)
    # February of both years; the average-annual block (MON = 2.0 years) is not a record.
    np.testing.assert_allclose(dataset["PRECIP"].values[1], [28.0, 56.0])
    np.testing.assert_allclose(dataset["PRECIP"].values[13], [28.0, 56.0])
    assert float(dataset["PRECIP"].max()) == 62.0
    assert dataset.attrs["swat_print_frequency"] == "monthly"
    assert dataset.attrs["swat_record_blocks"] == 24
    assert dataset.attrs["swat_annual_summary_blocks"] == 2
    assert dataset.attrs["swat_average_annual_summary_blocks"] == 1
    assert dataset.attrs["swat_average_annual_years"] == 2.0
    np.testing.assert_allclose(dataset["AREA"].values[0], [10.7, 5.2])
    np.testing.assert_array_equal(dataset["hru"].values, [1, 2])
    np.testing.assert_array_equal(dataset["GIS"].values, ["000010001", "000010002"])
    np.testing.assert_array_equal(dataset["LULC"].values[0], ["AGRL", "FRSD"])
    for name, units in (
        ("PRECIP", "mm"),
        ("ET", "mm"),
        ("SURQ_GEN", "mm"),
        ("LATQGEN", "mm"),
        ("TMP_MX", "degC"),
        ("SOL_TMP", "degC"),
        ("SOLAR", "MJ m-2"),
        ("AREA", "km2"),
    ):
        assert dataset[name].attrs["units"] == units, name
        assert dataset[name].attrs["units_source"] == "file", name
    # LAI has no unit suffix in heds: the adapter table supplies m2/m2 and says so.
    assert dataset["LAI"].attrs["units_source"] == "adapter_table"


def test_daily_output_hru_variant(tmp_path: Path) -> None:
    """IPRINT=1: julian-day blocks only; SWAT writes no annual or average-annual block (writea.f90, writeaa.f90)."""
    blocks = [_block(f"{364:4d}", 1.0), _block(f"{365:4d}", 1.0), _block(f"{1:4d}", 1.0), _block(f"{2:4d}", 1.0)]
    _write_output_hru(tmp_path, blocks)
    daily = dict(_MONTHLY, iprint=1, idaf=364, idal=2)
    adapter = _open(tmp_path, **daily)
    dataset = adapter.current_datasets["hru"]
    np.testing.assert_array_equal(dataset["MON"].values, [364, 365, 1, 2])
    np.testing.assert_array_equal(dataset["YR"].values, [2021, 2021, 2022, 2022])
    np.testing.assert_allclose(dataset["PRECIP"].values, [[1.0, 2.0]] * 4)
    np.testing.assert_allclose(dataset["time_bnds"].values, [[363.0, 364.0], [364.0, 365.0], [365.0, 366.0], [366.0, 367.0]])
    assert dataset.attrs["swat_print_frequency"] == "daily"
    assert dataset.attrs["swat_annual_summary_blocks"] == 0
    assert dataset.attrs["swat_average_annual_summary_blocks"] == 0
    flux = adapter.read(variables=["precipitation_flux"], dataset="hru")["precipitation_flux"]
    torch.testing.assert_close(flux, _flux_per_hru(4), rtol=1e-5, atol=1e-12)
    # A daily file carrying annual and average-annual blocks does not follow the SWAT source.
    with_summaries = blocks[:2] + [_block(f"{2021:4d}", 2.0)] + blocks[2:] + [_block(f"{2022:4d}", 2.0), _block(f"{2.0:4.1f}", 2.0)]
    _write_output_hru(tmp_path, with_summaries)
    with pytest.raises((ValueError, OSError), match="declared print sequence"):
        _open(tmp_path, **daily)


def test_annual_output_hru_records_are_the_annual_blocks(tmp_path: Path) -> None:
    """IPRINT=2: the annual blocks (here format 1001, HRU as i7) are the records; the average-annual block is excluded."""
    blocks = [_block(f"{year:4d}", 365.0, hru_width=7) for year in (2021, 2022)] + [_block(f"{2.0:4.1f}", 365.0, hru_width=7)]
    _write_output_hru(tmp_path, blocks)
    dataset = _open(tmp_path, **{**_MONTHLY, "iprint": 2}).current_datasets["hru"]
    np.testing.assert_array_equal(dataset["MON"].values, [2021, 2022])
    np.testing.assert_allclose(dataset["time_bnds"].values, [[0.0, 365.0], [365.0, 730.0]])
    assert dataset.attrs["swat_print_frequency"] == "annual"
    assert dataset.attrs["swat_annual_summary_blocks"] == 0
    assert dataset.attrs["swat_average_annual_summary_blocks"] == 1


def test_nyskip_warm_up_years_are_not_printed(tmp_path: Path) -> None:
    """NYSKIP=1: printing starts on January 1 of IYR + 1; IDAF applies only to the first simulated year."""
    _write_output_hru(tmp_path, _monthly_blocks(years=(2021, 2022)))
    run = dict(_MONTHLY, nbyr=3, iyr=2020, idaf=100, idal=0, nyskip=1)
    dataset = _open(tmp_path, **run).current_datasets["hru"]
    assert dataset.sizes["time"] == 24
    np.testing.assert_array_equal(dataset["YR"].values, [2021] * 12 + [2022] * 12)
    # 2020 is a leap year: January 2021 starts 366 days after 2020-01-01.
    np.testing.assert_allclose(dataset["time_bnds"].values[0], [366.0, 397.0])
    np.testing.assert_allclose(dataset["time_bnds"].values[-1], [1065.0, 1096.0])
    assert dataset.attrs["swat_nyskip"] == 1
    assert dataset.attrs["swat_annual_summary_blocks"] == 2
    # A file that prints the warm-up year does not follow the declared settings.
    _write_output_hru(tmp_path, _monthly_blocks(years=(2020, 2021, 2022)))
    with pytest.raises((ValueError, OSError), match="declared print sequence"):
        _open(tmp_path, **run)
    # Daily print: IDAF=364 of the skipped first year is ignored, printing starts on day 1 of 2022.
    _write_output_hru(tmp_path, [_block(f"{day:4d}", 1.0) for day in (1, 2, 3)])
    daily = dict(_MONTHLY, iprint=1, nbyr=2, iyr=2021, idaf=364, idal=3, nyskip=1)
    dataset = _open(tmp_path, **daily).current_datasets["hru"]
    np.testing.assert_array_equal(dataset["YR"].values, [2022, 2022, 2022])
    np.testing.assert_allclose(dataset["time_bnds"].values, [[365.0, 366.0], [366.0, 367.0], [367.0, 368.0]])
    # NYSKIP >= NBYR prints the last year only (readfile.f90 line 293: nyskip = nbyr - 1).
    dataset = _open(tmp_path, **dict(daily, nyskip=5)).current_datasets["hru"]
    assert dataset.attrs["swat_nyskip"] == 1
    assert "readfile.f90 line 293" in dataset.attrs["swat_print_settings_source"]


def test_idal_zero_is_the_last_day_and_the_calendar_must_match_swat(tmp_path: Path) -> None:
    """IDAL=0 ends the last year on its last day (simulate.f90); SWAT leap years are mod(year, 4) == 0."""
    _write_output_hru(tmp_path, _monthly_blocks())
    dataset = _open(tmp_path, **dict(_MONTHLY, idal=0)).current_datasets["hru"]
    np.testing.assert_allclose(dataset["time_bnds"].values[-1], [699.0, 730.0])
    _write_output_hru(tmp_path, _monthly_blocks(years=(2023, 2024)))
    with pytest.raises((ValueError, OSError), match="simulate.f90"):
        _open(tmp_path, **dict(_MONTHLY, iyr=2023, calendar="noleap"))


def _corrupted(kind: str) -> list[list[str]]:
    blocks = _monthly_blocks()
    if kind == "annual_before_its_months":
        blocks.insert(0, blocks.pop(12))
    elif kind == "no_average_annual_block":
        blocks.pop()
    elif kind == "period_after_the_last_annual_block":
        blocks.insert(len(blocks) - 1, _block(f"{1:4d}", 31.0))
    elif kind == "skipped_month":
        blocks.pop(2)
    elif kind == "repeated_month":
        blocks.insert(1, blocks[0])
    elif kind == "month_fitting_no_rule":
        blocks[4] = _block(f"{13:4d}", 31.0)
    else:
        raise AssertionError(kind)
    return blocks


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("annual_before_its_months", "declared print sequence"),
        ("no_average_annual_block", "ends after"),
        ("period_after_the_last_annual_block", "declared print sequence"),
        ("skipped_month", "declared print sequence"),
        ("repeated_month", "HRU"),
        ("month_fitting_no_rule", "fits no rule"),
    ],
)
def test_corrupted_block_sequence_raises(tmp_path: Path, kind: str, message: str) -> None:
    _write_output_hru(tmp_path, _corrupted(kind))
    with pytest.raises((ValueError, OSError), match=message):
        _open(tmp_path, **_MONTHLY)


def test_print_settings_are_declared_not_guessed(tmp_path: Path) -> None:
    _write_output_hru(tmp_path, _monthly_blocks())
    without_iprint = {key: value for key, value in _MONTHLY.items() if key != "iprint"}
    with pytest.raises((ValueError, OSError), match="IPRINT"):
        _open(tmp_path, **without_iprint)
    without_calendar = {key: value for key, value in _MONTHLY.items() if key != "calendar"}
    with pytest.raises((ValueError, OSError), match="calendar"):
        _open(tmp_path, **without_calendar)
    with pytest.raises((ValueError, OSError), match="ICALEN=1"):
        _open(tmp_path, **{**_MONTHLY, "icalen": 1})
    with pytest.raises(ValueError, match="iprint"):
        SWATConfig(root_path=tmp_path, output_level="hru", iprint=3)


@pytest.mark.parametrize("labels", [True, False], ids=["labelled", "unlabelled"])
def test_file_cio_is_read_by_replaying_the_readfile_sequence(tmp_path: Path, labels: bool) -> None:
    """SWAT reads file.cio by position; the writer labels are optional and never used to locate values."""
    _write_output_hru(tmp_path, _monthly_blocks())
    _write_file_cio(tmp_path, _file_cio_lines(labels=labels))
    dataset = _open(tmp_path, calendar="standard").current_datasets["hru"]
    assert dataset.sizes["time"] == 24
    source = dataset.attrs["swat_print_settings_source"]
    for name, line in (("NBYR", 8), ("IYR", 9), ("IDAF", 10), ("IDAL", 11), ("IPRINT", 59), ("NYSKIP", 60), ("ICALEN", 85)):
        assert f"{name}=file.cio line {line}" in source, (name, source)
    assert "calendar=configuration" in source


def test_file_cio_list_directed_reads_span_records_and_skip_blank_records(tmp_path: Path) -> None:
    """A list-directed read takes values across records and skips blank ones (Fortran 10.10.3)."""
    _write_output_hru(tmp_path, _monthly_blocks())
    lines = _file_cio_lines()
    lines[64:65] = ["   0" * 10, "   0" * 10]  # IPDVAR over two records
    lines.insert(58, "")  # blank record before IPRINT
    _write_file_cio(tmp_path, lines)
    source = _open(tmp_path, calendar="standard").current_datasets["hru"].attrs["swat_print_settings_source"]
    assert "IPRINT=file.cio line 60" in source
    assert "NYSKIP=file.cio line 61" in source
    assert "ICALEN=file.cio line 87" in source


def test_file_cio_without_the_icalen_record_keeps_the_swat_initial_value(tmp_path: Path) -> None:
    """readfile.f90 line 740 reads ICALEN with iostat=eof; zero0.f90 initialises icalen = 0."""
    _write_output_hru(tmp_path, _monthly_blocks())
    _write_file_cio(tmp_path, _file_cio_lines()[:84])
    dataset = _open(tmp_path, calendar="standard").current_datasets["hru"]
    assert dataset.attrs["swat_icalen"] == 0
    assert "zero0.f90" in dataset.attrs["swat_print_settings_source"]


def test_file_cio_labels_positions_and_configuration_are_cross_checked(tmp_path: Path) -> None:
    _write_output_hru(tmp_path, _monthly_blocks())
    lines = _file_cio_lines()
    lines[58], lines[59] = lines[58].replace("IPRINT", "NYSKIP"), lines[59].replace("NYSKIP", "IPRINT")
    _write_file_cio(tmp_path, lines)
    with pytest.raises((ValueError, OSError), match="label"):
        _open(tmp_path, calendar="standard")
    shifted = _file_cio_lines()
    del shifted[12]  # one climate record missing: every later read moves up one record
    _write_file_cio(tmp_path, shifted)
    with pytest.raises((ValueError, OSError), match="file.cio line"):
        _open(tmp_path, calendar="standard")
    _write_file_cio(tmp_path, _file_cio_lines()[:50])
    with pytest.raises((ValueError, OSError), match="end of file"):
        _open(tmp_path, calendar="standard")
    _write_file_cio(tmp_path, _file_cio_lines())
    with pytest.raises((ValueError, OSError), match="file.cio"):
        _open(tmp_path, calendar="standard", iyr=2020)


def test_malformed_output_hru_raises_instead_of_being_skipped(tmp_path: Path) -> None:
    blocks = _monthly_blocks()
    blocks[3] = [blocks[3][0][:-10], blocks[3][1]]
    _write_output_hru(tmp_path, blocks)
    with pytest.raises((ValueError, OSError), match="line"):
        _open(tmp_path, **_MONTHLY)
    blocks = _monthly_blocks()
    blocks[3] = [blocks[3][0][:-10] + "**********", blocks[3][1]]
    _write_output_hru(tmp_path, blocks)
    with pytest.raises((ValueError, OSError), match="line"):
        _open(tmp_path, **_MONTHLY)
    _write_output_hru(tmp_path, _monthly_blocks(), header=_HEADER.replace("LULC  HRU", "LULC HRU "))
    with pytest.raises((ValueError, OSError), match="header"):
        _open(tmp_path, **_MONTHLY)
    calendar_header = "LULC  HRU       GIS  SUB  MGT MO DA   YR   AREAkm2" + "".join(_HEDS)
    _write_output_hru(tmp_path, _monthly_blocks(), header=calendar_header)
    with pytest.raises((ValueError, OSError), match="1021"):
        _open(tmp_path, **_MONTHLY)
    _write_output_hru(tmp_path, [], header="  no swat header here")
    with pytest.raises((ValueError, OSError), match="header"):
        _open(tmp_path, **_MONTHLY)


def test_monthly_record_bounds_are_calendar_months_from_the_declared_start(tmp_path: Path) -> None:
    """IPRINT=0 records are calendar months from IYR/IDAF: February 28 mm and March 31 mm are both 1 mm day-1."""
    _write_output_hru(tmp_path, _monthly_blocks(first_day=10))
    config = {**_MONTHLY, "idaf": 10}
    adapter = _open(tmp_path, **config)
    dataset = adapter.current_datasets["hru"]
    assert dataset["time"].attrs["units"] == "days since 2021-01-01 00:00:00"
    assert dataset["time"].attrs["bounds"] == "time_bnds"
    expected_days = [calendar.monthrange(year, month)[1] for year in (2021, 2022) for month in range(1, 13)]
    expected_days[0] -= 9
    np.testing.assert_allclose(np.diff(dataset["time_bnds"].values, axis=1)[:, 0], expected_days)
    assert float(dataset["time_bnds"].values[0, 0]) == 9.0
    np.testing.assert_allclose(dataset["PRECIP"].values[1:3, 0], [28.0, 31.0])
    flux = adapter.read(variables=["precipitation_flux"], dataset="hru")["precipitation_flux"]
    torch.testing.assert_close(flux[1:3], _flux_per_hru(2), rtol=1e-5, atol=1e-12)
    torch.testing.assert_close(flux, _flux_per_hru(24), rtol=1e-5, atol=1e-12)
    assert "time bounds" in flux._relic_meta["attrs"]["flux_interval_source"]
    # A fixed record interval is not a declaration of calendar-month records.
    with pytest.raises((ValueError, OSError), match="record_interval"):
        _open(tmp_path, **config, record_interval=30.0, record_interval_units="day")


def test_only_the_verified_output_hru_layout_is_read(tmp_path: Path) -> None:
    """SWATConfig declares its output level; only the output.hru text layout is implemented, and the
    unverified SQLite path (every table loaded without a layout check) is gone."""
    import dataclasses

    with pytest.raises(TypeError, match="output_level"):
        SWATConfig(root_path=tmp_path)
    for level in ("subbasin", "reach", "reservoir"):
        with pytest.raises(ValueError, match="output.hru"):
            SWATConfig(root_path=tmp_path, output_level=level)
    fields = {field.name for field in dataclasses.fields(SWATConfig)}
    assert "output_format" not in fields and "database_path" not in fields
    assert not hasattr(SWATAdapter, "_open_sqlite_files")
    assert SWATConfig(root_path=tmp_path, output_level="hru").output_level == "hru"

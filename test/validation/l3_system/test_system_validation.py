"""
L3 System Tests: Complete System Validation and Real-World Testing

Tests for complete system validation, real-world data integration,
scientific validation against observational datasets, and production readiness.

Focus Areas:
1. Real-World Dataset Integration (FluxNet, MODIS, etc.)
2. Multi-Scale Validation (Leaf 鈫?Canopy 鈫?Ecosystem 鈫?Regional)
3. Temporal Validation (Diurnal, Seasonal, Inter-annual)
4. Scientific Process Validation
5. Production System Readiness

Mathematical Framework:
    Multi-scale validation hierarchy:
    - Leaf scale: A鈧椻倯鈧?f (Ci, T鈧椻倯鈧?f, VPD, PAR)
    - Canopy scale: GPPcanopy = 鈭?A鈧椻倯鈧?f 脳 LAI(z) 脳 f_light(z) dz
    - Ecosystem scale: NEE = GPP - R鈧?co - R鈧?鈧掅耽鈧?    - Regional scale: 鈭戓耽 NEE岬?脳 Area岬?/ 鈭戓耽 Area岬?    
    Temporal validation:
    - Diurnal patterns: f(hour) with expected photosynthetic/respiratory cycles
    - Seasonal patterns: f(DOY) with phenological and climatic drivers
    - Inter-annual: f(year) with climate variability and trends
    
References:
    - Baldocchi (2003): Assessing the eddy covariance technique for evaluating CO2 exchange
    - Jung et al. (2020): Scaling carbon fluxes from eddy covariance sites to globe
    - Pastorello et al. (2020): The FLUXNET2015 dataset
    - Running et al. (2004): A continuous satellite-derived measure of global terrestrial primary production
"""

import pytest
import numpy as np
from typing import Dict, Any
from datetime import datetime, timedelta

from relic.validation.framework.models import (
    ValidationLevel, ValidationCode, ValidationResultModel, ValidationContext
)
from relic.validation.framework.engine import ValidationEngine
from relic.validation.framework.decorators import validated_rule


class TestRealWorldDatasetIntegration:
    """Test integration with real-world scientific datasets."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ValidationEngine()
        
        # Create mock real-world datasets
        self.fluxnet_data = self._create_mock_fluxnet_data()
        self.modis_data = self._create_mock_modis_data()
        self.climate_data = self._create_mock_climate_data()
    
    def _create_mock_fluxnet_data(self) -> Dict[str, Any]:
        """Create mock FluxNet-style eddy covariance data."""
        np.random.seed(100)  # Reproducible mock data
        
        # Create 2 years of half-hourly data
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(minutes=30*i) for i in range(35040)]  # 2 years 脳 365 days 脳 48 half-hours
        
        # Generate realistic diurnal and seasonal patterns
        hour_of_day = np.array([d.hour + d.minute/60.0 for d in dates])
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        
        # Seasonal temperature pattern (掳C)
        temp_seasonal = 15 + 10 * np.cos(2 * np.pi * (day_of_year - 172) / 365)  # Peak in summer
        temp_diurnal = 5 * np.cos(2 * np.pi * (hour_of_day - 14) / 24)  # Peak at 2 PM
        temperature = temp_seasonal + temp_diurnal + np.random.normal(0, 2, len(dates))
        
        # PAR with realistic diurnal pattern (渭mol m-2 s-1)
        par_max = 2000 * (0.7 + 0.3 * np.cos(2 * np.pi * (day_of_year - 172) / 365))  # Seasonal variation
        par_diurnal = par_max * np.maximum(0, np.cos(2 * np.pi * (hour_of_day - 12) / 24))
        par_noise = np.random.normal(0, 50, len(dates))
        par = np.maximum(0, par_diurnal + par_noise)
        
        # GPP with temperature and light dependencies (渭mol CO2 m-2 s-1)
        temp_factor = np.exp(0.1 * (temperature - 25)) / (1 + np.exp(0.5 * (temperature - 35)))
        light_factor = par / (par + 500)  # Michaelis-Menten response
        gpp_potential = 30 * temp_factor * light_factor
        gpp = np.maximum(0, gpp_potential + np.random.normal(0, 2, len(dates)))
        
        # NEE (Net Ecosystem Exchange) - negative values are uptake
        respiration_base = 5 * np.exp(0.05 * temperature)
        nee = respiration_base - gpp + np.random.normal(0, 1, len(dates))
        
        # Evapotranspiration (mm day-1 converted to half-hourly)
        et_daily = 2 + 3 * np.sin(2 * np.pi * (day_of_year - 100) / 365)  # Peak in late spring/summer
        vpd = np.maximum(0.1, 2 + np.sin(2 * np.pi * (day_of_year - 150) / 365) + np.random.normal(0, 0.3, len(dates)))
        et = (et_daily / 48) * (1 + vpd/5) * np.maximum(0, np.sin(2 * np.pi * hour_of_day / 24)) + np.random.normal(0, 0.01, len(dates))
        et = np.maximum(0, et)
        
        return {
            'datetime': dates,
            'air_temperature': temperature,  # 掳C
            'par': par,  # 渭mol m-2 s-1
            'gpp': gpp,  # 渭mol CO2 m-2 s-1
            'nee': nee,  # 渭mol CO2 m-2 s-1
            'et': et,  # mm per 30 min
            'vpd': vpd,  # kPa
            'site_id': 'TEST_SITE',
            'latitude': 45.5,
            'longitude': -120.8,
            'elevation': 1200  # meters
        }
    
    def _create_mock_modis_data(self) -> Dict[str, Any]:
        """Create mock MODIS-style satellite data."""
        np.random.seed(101)
        
        # Monthly data for 2 years
        months = []
        start_date = datetime(2020, 1, 1)
        for i in range(24):  # 2 years 脳 12 months
            months.append(start_date.replace(month=(i % 12) + 1, year=start_date.year + (i // 12)))
        
        # LAI with seasonal pattern
        month_of_year = np.array([(d.month - 1) for d in months])
        lai_seasonal = 2 + 1.5 * np.cos(2 * np.pi * (month_of_year - 5) / 12)  # Peak in June
        lai = np.maximum(0.5, lai_seasonal + np.random.normal(0, 0.3, len(months)))
        
        # NDVI correlated with LAI
        ndvi = 0.3 + 0.4 * (lai - 0.5) / 3.0 + np.random.normal(0, 0.05, len(months))
        ndvi = np.clip(ndvi, 0, 1)
        
        # GPP from satellite (g C m-2 day-1)
        gpp_satellite = 5 + 10 * lai / 4.0 + np.random.normal(0, 1, len(months))
        gpp_satellite = np.maximum(0, gpp_satellite)
        
        return {
            'datetime': months,
            'lai': lai,  # m2 m-2
            'ndvi': ndvi,  # dimensionless
            'gpp_satellite': gpp_satellite,  # g C m-2 day-1
            'pixel_size': 500,  # meters
            'quality_flag': np.random.choice(['good', 'fair', 'poor'], len(months), p=[0.7, 0.2, 0.1])
        }
    
    def _create_mock_climate_data(self) -> Dict[str, Any]:
        """Create mock climate data."""
        np.random.seed(102)
        
        # Daily data for 2 years
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(731)]  # 2 years + leap day
        
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        
        # Precipitation with stochastic events
        precip_base = 2 + np.sin(2 * np.pi * (day_of_year - 100) / 365)
        precip_events = np.random.poisson(0.3, len(dates)) * np.random.exponential(5, len(dates))
        precipitation = precip_base + precip_events
        
        # Radiation (MJ m-2 day-1)  
        radiation_max = 25 + 5 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
        radiation = radiation_max + np.random.normal(0, 2, len(dates))
        radiation = np.maximum(5, radiation)
        
        return {
            'datetime': dates,
            'precipitation': precipitation,  # mm day-1
            'solar_radiation': radiation,  # MJ m-2 day-1
            'data_source': 'GRIDDED_CLIMATE'
        }
    
    def test_fluxnet_integration_validation(self):
        """Test integration and validation against FluxNet data."""
        @validated_rule(
            rule_id="fluxnet_integration_validation",
            layer="L3",
            domains=["biosphere", "atmosphere"],
            tags=["fluxnet", "eddy_covariance", "integration"]
        )
        def validate_fluxnet_integration(context):
            """Validate model integration against FluxNet eddy covariance data."""
            results = []
            
            fluxnet_data = context.config.get('fluxnet_data', {})
            model_data = context.config.get('model_data', {})
            
            if not fluxnet_data or not model_data:
                return [ValidationResultModel(
                    level=ValidationLevel.ERROR,
                    code=ValidationCode.MISSING_REQUIRED_DATA,
                    message="Missing FluxNet or model data for integration validation",
                    layer="L3",
                    scope="fluxnet_integration_validation"
                )]
            
            # Temporal alignment validation
            obs_timestamps = fluxnet_data.get('datetime', [])
            model_timestamps = model_data.get('datetime', [])
            
            # Check temporal coverage
            if len(obs_timestamps) != len(model_timestamps):
                results.append(ValidationResultModel(
                    level=ValidationLevel.WARNING,
                    code=ValidationCode.GENERIC_VALIDATION_FAILED,
                    message=f"Temporal mismatch: FluxNet {len(obs_timestamps)} vs Model {len(model_timestamps)} timesteps",
                    layer="L3",
                    scope="fluxnet_integration_validation"
                ))
            
            # Validate key carbon fluxes
            variables_to_validate = ['gpp', 'nee', 'et']
            
            for var in variables_to_validate:
                if var in fluxnet_data and var in model_data:
                    obs_values = np.array(fluxnet_data[var])
                    model_values = np.array(model_data[var])
                    
                    # Basic quality control - remove outliers and gaps
                    valid_mask = (
                        np.isfinite(obs_values) & np.isfinite(model_values) &
                        (obs_values > -100) & (obs_values < 100) &  # Reasonable range for fluxes
                        (model_values > -100) & (model_values < 100)
                    )
                    
                    if np.sum(valid_mask) < 0.7 * len(obs_values):
                        results.append(ValidationResultModel(
                            level=ValidationLevel.WARNING,
                            code=ValidationCode.GENERIC_VALIDATION_FAILED,
                            message=f"Variable {var}: High data gap rate ({np.sum(valid_mask)/len(obs_values):.1%} valid)",
                            layer="L3",
                            scope="fluxnet_integration_validation"
                        ))
                        continue
                    
                    obs_clean = obs_values[valid_mask]
                    model_clean = model_values[valid_mask]
                    
                    # Calculate validation statistics (manual correlation to avoid numpy.corrcoef issues)
                    correlation = 0.0
                    if len(obs_clean) > 1:
                        try:
                            # Manual correlation calculation (Pearson correlation coefficient)
                            obs_mean = np.mean(obs_clean)
                            model_mean = np.mean(model_clean)
                            
                            obs_centered = obs_clean - obs_mean
                            model_centered = model_clean - model_mean
                            
                            numerator = np.sum(obs_centered * model_centered)
                            obs_ss = np.sum(obs_centered ** 2)
                            model_ss = np.sum(model_centered ** 2)
                            
                            # Check for non-zero sum of squares
                            if obs_ss > 1e-10 and model_ss > 1e-10:
                                correlation = numerator / np.sqrt(obs_ss * model_ss)
                                # Ensure correlation is within valid range
                                correlation = np.clip(correlation, -1.0, 1.0)
                                if not np.isfinite(correlation):
                                    correlation = 0.0
                        except Exception:
                            correlation = 0.0
                    rmse = np.sqrt(np.mean((model_clean - obs_clean) ** 2))
                    bias = np.mean(model_clean - obs_clean)
                    relative_bias = bias / np.mean(np.abs(obs_clean)) if np.mean(np.abs(obs_clean)) > 0 else float('inf')
                    
                    # Validate against FluxNet quality standards
                    if var == 'gpp':
                        # GPP validation thresholds
                        if correlation > 0.8 and abs(relative_bias) < 0.2:
                            level = ValidationLevel.INFO
                            message = f"GPP validation excellent: r={correlation:.3f}, bias={relative_bias:.1%}"
                        elif correlation > 0.6 and abs(relative_bias) < 0.4:
                            level = ValidationLevel.WARNING
                            message = f"GPP validation acceptable: r={correlation:.3f}, bias={relative_bias:.1%}"
                        else:
                            level = ValidationLevel.ERROR
                            message = f"GPP validation poor: r={correlation:.3f}, bias={relative_bias:.1%}"
                    
                    elif var == 'nee':
                        # NEE validation (more challenging due to small signal)
                        if correlation > 0.5 and abs(relative_bias) < 0.5:
                            level = ValidationLevel.INFO
                            message = f"NEE validation good: r={correlation:.3f}, bias={relative_bias:.1%}"
                        elif correlation > 0.3 and abs(relative_bias) < 1.0:
                            level = ValidationLevel.WARNING
                            message = f"NEE validation acceptable: r={correlation:.3f}, bias={relative_bias:.1%}"
                        else:
                            level = ValidationLevel.ERROR
                            message = f"NEE validation poor: r={correlation:.3f}, bias={relative_bias:.1%}"
                    
                    elif var == 'et':
                        # ET validation
                        if correlation > 0.7 and abs(relative_bias) < 0.3:
                            level = ValidationLevel.INFO
                            message = f"ET validation good: r={correlation:.3f}, bias={relative_bias:.1%}"
                        elif correlation > 0.5 and abs(relative_bias) < 0.5:
                            level = ValidationLevel.WARNING
                            message = f"ET validation acceptable: r={correlation:.3f}, bias={relative_bias:.1%}"
                        else:
                            level = ValidationLevel.ERROR
                            message = f"ET validation poor: r={correlation:.3f}, bias={relative_bias:.1%}"
                    
                    result = ValidationResultModel(
                        level=level,
                        code=ValidationCode.GENERIC_VALIDATION_FAILED,  # TODO: Add FLUXNET_VALIDATION codes
                        message=message,
                        layer="L3",
                        scope="fluxnet_integration_validation",
                        tags=["fluxnet", var, "eddy_covariance"],
                        metrics={
                            "correlation": correlation,
                            "rmse": rmse,
                            "bias": bias,
                            "relative_bias": relative_bias,
                            "n_valid_points": int(np.sum(valid_mask)),
                            "data_completeness": float(np.sum(valid_mask) / len(obs_values))
                        }
                    )
                    
                    # Add FluxNet validation detail
                    result.add_detail(
                        variable_name=var,
                        cf_standard_name=f"{var}_flux_validation",
                        units="渭mol m-2 s-1" if var in ['gpp', 'nee'] else "mm per 30min",
                        value=correlation,
                        threshold=0.6 if var == 'gpp' else 0.3,
                        statistics={
                            "observed_mean": float(np.mean(obs_clean)),
                            "modeled_mean": float(np.mean(model_clean)),
                            "observed_std": float(np.std(obs_clean)),
                            "modeled_std": float(np.std(model_clean)),
                            "site_info": {
                                "site_id": fluxnet_data.get('site_id', 'UNKNOWN'),
                                "latitude": fluxnet_data.get('latitude', 0.0),
                                "longitude": fluxnet_data.get('longitude', 0.0),
                                "elevation": fluxnet_data.get('elevation', 0.0)
                            }
                        }
                    )
                    
                    results.append(result)
            
            return results
        
        # Create corresponding model data with realistic performance
        model_gpp = np.array(self.fluxnet_data['gpp']) * 0.95 + np.random.normal(0, 1.0, len(self.fluxnet_data['gpp']))
        model_nee = np.array(self.fluxnet_data['nee']) * 1.1 + np.random.normal(0, 0.5, len(self.fluxnet_data['nee']))
        model_et = np.array(self.fluxnet_data['et']) * 0.9 + np.random.normal(0, 0.02, len(self.fluxnet_data['et']))
        
        model_data = {
            'datetime': self.fluxnet_data['datetime'],
            'gpp': model_gpp,
            'nee': model_nee, 
            'et': model_et
        }
        
        test_context = ValidationContext(config={
            'fluxnet_data': self.fluxnet_data,
            'model_data': model_data
        })
        
        results = validate_fluxnet_integration(test_context)
        
        # Should have results for all three variables
        assert len(results) >= 3
        
        # Find GPP result
        gpp_result = next((r for r in results if 'gpp' in r.tags), None)
        assert gpp_result is not None
        assert gpp_result.metrics['correlation'] > 0.8  # Should be high correlation
        assert gpp_result.level == ValidationLevel.INFO  # Should be excellent
        
        # Check FluxNet site information is included
        assert len(gpp_result.details) == 1
        detail = gpp_result.details[0]
        assert 'site_info' in detail.statistics
        assert detail.statistics['site_info']['site_id'] == 'TEST_SITE'


class TestMultiScaleValidation:
    """Test multi-scale validation from leaf to ecosystem level."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ValidationEngine()
    
    def test_leaf_to_canopy_scaling_validation(self):
        """Test validation of leaf-to-canopy scaling relationships."""
        @validated_rule(
            rule_id="leaf_canopy_scaling_validation",
            layer="L3",
            domains=["biosphere"],
            tags=["scaling", "canopy", "photosynthesis"]
        )
        def validate_leaf_canopy_scaling(context):
            """Validate leaf-to-canopy scaling using big-leaf vs. multi-layer approaches."""
            results = []
            
            scaling_data = context.config.get('scaling_data', {})
            
            # Extract scaling components
            leaf_photosynthesis = scaling_data.get('leaf_photosynthesis_max', 0.0)  # 渭mol CO2 m-2 s-1
            lai = scaling_data.get('lai', 1.0)  # m2 m-2
            extinction_coefficient = scaling_data.get('extinction_coefficient', 0.5)
            canopy_photosynthesis_observed = scaling_data.get('canopy_photosynthesis', 0.0)
            
            # Calculate theoretical canopy photosynthesis using Beer-Lambert law
            # Simplified big-leaf approach: Ac = Amax 脳 LAI 脳 (1 - exp(-k脳LAI))/(k脳LAI)
            if lai > 0 and extinction_coefficient > 0:
                scaling_efficiency = (1 - np.exp(-extinction_coefficient * lai)) / (extinction_coefficient * lai)
                canopy_photosynthesis_theoretical = leaf_photosynthesis * lai * scaling_efficiency
            else:
                canopy_photosynthesis_theoretical = 0.0
            
            # Validate scaling relationship
            if canopy_photosynthesis_theoretical > 0:
                scaling_error = abs(canopy_photosynthesis_observed - canopy_photosynthesis_theoretical) / canopy_photosynthesis_theoretical
                
                if scaling_error < 0.15:  # Within 15%
                    level = ValidationLevel.INFO
                    message = f"Leaf-canopy scaling excellent: {scaling_error:.1%} error"
                elif scaling_error < 0.3:  # Within 30%
                    level = ValidationLevel.WARNING
                    message = f"Leaf-canopy scaling acceptable: {scaling_error:.1%} error"
                else:
                    level = ValidationLevel.ERROR
                    message = f"Leaf-canopy scaling poor: {scaling_error:.1%} error"
            else:
                level = ValidationLevel.ERROR
                message = "Leaf-canopy scaling calculation failed: theoretical photosynthesis is zero"
                scaling_error = float('inf')
            
            result = ValidationResultModel(
                level=level,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,  # TODO: Add SCALING_VALIDATION codes
                message=message,
                layer="L3",
                scope="leaf_canopy_scaling_validation",
                tags=["scaling", "big_leaf", "beer_lambert"],
                metrics={
                    "scaling_error": scaling_error,
                    "scaling_efficiency": scaling_efficiency,
                    "theoretical_canopy_ps": canopy_photosynthesis_theoretical,
                    "observed_canopy_ps": canopy_photosynthesis_observed
                }
            )
            
            # Add scaling detail
            result.add_detail(
                variable_name="canopy_photosynthesis_scaling",
                cf_standard_name="canopy_photosynthesis_scaling_validation",
                units="渭mol CO2 m-2 s-1",
                value=canopy_photosynthesis_observed,
                expected=canopy_photosynthesis_theoretical,
                threshold=canopy_photosynthesis_theoretical * 0.15,
                statistics={
                    "leaf_photosynthesis_max": leaf_photosynthesis,
                    "leaf_area_index": lai,
                    "light_extinction_coefficient": extinction_coefficient,
                    "scaling_efficiency_factor": scaling_efficiency,
                    "upscaling_method": "beer_lambert_big_leaf"
                }
            )
            
            results.append(result)
            return results
        
        # Test realistic scaling scenario
        scaling_scenario = {
            'scaling_data': {
                'leaf_photosynthesis_max': 25.0,  # 渭mol CO2 m-2 s-1
                'lai': 4.5,  # m2 m-2
                'extinction_coefficient': 0.5,
                'canopy_photosynthesis': 45.0  # 渭mol CO2 m-2 s-1 (close to theoretical ~44.7)
            }
        }
        
        test_context = ValidationContext(config=scaling_scenario)
        results = validate_leaf_canopy_scaling(test_context)
        
        assert len(results) == 1
        result = results[0]
        
        # Should validate good scaling
        assert result.level == ValidationLevel.INFO
        assert result.metrics['scaling_error'] < 0.15
        
        # Check scaling calculation
        theoretical = result.metrics['theoretical_canopy_ps']
        assert theoretical > 40.0  # Should be around 44.7 based on Beer-Lambert
        assert theoretical < 50.0
        
        # Check detail
        detail = result.details[0]
        assert detail.variable_name == "canopy_photosynthesis_scaling"
        assert "upscaling_method" in detail.statistics
    
    def test_ecosystem_level_validation(self):
        """Test ecosystem-level carbon balance validation."""
        @validated_rule(
            rule_id="ecosystem_carbon_balance",
            layer="L3",
            domains=["biosphere"],
            tags=["ecosystem", "carbon_balance", "nep"]
        )
        def validate_ecosystem_carbon_balance(context):
            """Validate ecosystem-level carbon balance and NEP calculation."""
            results = []
            
            ecosystem_data = context.config.get('ecosystem_data', {})
            
            # Extract ecosystem carbon fluxes
            gpp = ecosystem_data.get('gpp_annual', 0.0)  # kg C m-2 yr-1
            ra = ecosystem_data.get('autotrophic_respiration_annual', 0.0)  # kg C m-2 yr-1
            rh = ecosystem_data.get('heterotrophic_respiration_annual', 0.0)  # kg C m-2 yr-1
            
            # Calculate derived quantities
            npp = gpp - ra  # Net Primary Production
            nep = gpp - ra - rh  # Net Ecosystem Production
            cue = ra / gpp if gpp > 0 else 0.0  # Carbon Use Efficiency
            
            # Validate against typical ecosystem values
            validation_issues = []
            
            # GPP should be reasonable for terrestrial ecosystems (0.5-3.0 kg C m-2 yr-1)
            if gpp < 0.5 or gpp > 3.0:
                validation_issues.append(f"GPP {gpp:.2f} kg C m-2 yr-1 outside typical range [0.5, 3.0]")
            
            # NPP should be 30-60% of GPP
            if gpp > 0:
                npp_fraction = npp / gpp
                if npp_fraction < 0.3 or npp_fraction > 0.7:
                    validation_issues.append(f"NPP/GPP ratio {npp_fraction:.2f} outside typical range [0.3, 0.7]")
            
            # Carbon Use Efficiency (Ra/GPP) should be 40-70%
            if cue < 0.3 or cue > 0.7:
                validation_issues.append(f"CUE {cue:.2f} outside typical range [0.3, 0.7]")
            
            # NEP should be small relative to GPP (ecosystem close to balance)
            if gpp > 0:
                nep_fraction = abs(nep) / gpp
                if nep_fraction > 0.2:  # NEP should be <20% of GPP for mature ecosystems
                    validation_issues.append(f"NEP/GPP ratio {nep_fraction:.2f} indicates large ecosystem imbalance")
            
            # Determine validation result
            if not validation_issues:
                level = ValidationLevel.INFO
                message = f"Ecosystem carbon balance realistic: GPP={gpp:.2f}, NPP/GPP={npp_fraction:.2f}, CUE={cue:.2f}"
            elif len(validation_issues) <= 2:
                level = ValidationLevel.WARNING
                message = f"Ecosystem carbon balance issues: {'; '.join(validation_issues[:2])}"
            else:
                level = ValidationLevel.ERROR
                message = f"Ecosystem carbon balance unrealistic: {'; '.join(validation_issues[:3])}"
            
            result = ValidationResultModel(
                level=level,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,  # TODO: Add ECOSYSTEM_BALANCE codes
                message=message,
                layer="L3",
                scope="ecosystem_carbon_balance",
                tags=["ecosystem", "carbon", "nep", "cue"],
                metrics={
                    "gpp_annual": gpp,
                    "npp_annual": npp,
                    "nep_annual": nep,
                    "carbon_use_efficiency": cue,
                    "npp_gpp_ratio": npp_fraction if gpp > 0 else 0.0,
                    "nep_gpp_ratio": nep_fraction if gpp > 0 else 0.0,
                    "validation_issues_count": len(validation_issues)
                }
            )
            
            # Add ecosystem balance detail
            result.add_detail(
                variable_name="ecosystem_carbon_balance",
                cf_standard_name="ecosystem_carbon_balance_validation",
                units="kg C m-2 yr-1",
                value=nep,
                statistics={
                    "gross_primary_production": gpp,
                    "autotrophic_respiration": ra,
                    "heterotrophic_respiration": rh,
                    "net_primary_production": npp,
                    "net_ecosystem_production": nep,
                    "carbon_use_efficiency": cue,
                    "ecosystem_type": "temperate_forest",  # Could be parameterized
                    "validation_issues": validation_issues
                }
            )
            
            results.append(result)
            return results
        
        # Test realistic ecosystem carbon balance
        realistic_ecosystem = {
            'ecosystem_data': {
                'gpp_annual': 1.8,  # kg C m-2 yr-1 (typical for temperate forest)
                'autotrophic_respiration_annual': 0.9,  # 50% of GPP
                'heterotrophic_respiration_annual': 0.8  # Small positive NEP (carbon sink)
            }
        }
        
        test_context = ValidationContext(config=realistic_ecosystem)
        results = validate_ecosystem_carbon_balance(test_context)
        
        assert len(results) == 1
        result = results[0]
        
        # Should validate as realistic ecosystem
        assert result.level == ValidationLevel.INFO
        assert result.metrics['carbon_use_efficiency'] == pytest.approx(0.5, rel=0.1)
        assert result.metrics['npp_gpp_ratio'] == pytest.approx(0.5, rel=0.1)
        assert result.metrics['nep_annual'] == pytest.approx(0.1, rel=0.1)  # Small carbon sink
        
        # Check ecosystem detail
        detail = result.details[0]
        assert detail.variable_name == "ecosystem_carbon_balance"
        assert len(detail.statistics['validation_issues']) == 0  # No issues


class TestTemporalValidation:
    """Test temporal pattern validation (diurnal, seasonal, inter-annual)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ValidationEngine()
    
    def test_diurnal_pattern_validation(self):
        """Test validation of diurnal patterns in carbon and water fluxes."""
        @validated_rule(
            rule_id="diurnal_pattern_validation",
            layer="L3",
            domains=["biosphere"],
            tags=["temporal", "diurnal", "patterns"]
        )
        def validate_diurnal_patterns(context):
            """Validate diurnal patterns in photosynthesis and evapotranspiration."""
            results = []
            
            diurnal_data = context.config.get('diurnal_data', {})
            
            # Extract hourly data
            hours = diurnal_data.get('hour', [])
            gpp = diurnal_data.get('gpp', [])  # 渭mol CO2 m-2 s-1
            et = diurnal_data.get('et', [])  # mm hr-1
            par = diurnal_data.get('par', [])  # 渭mol m-2 s-1
            
            if len(hours) != 24 or len(gpp) != 24 or len(et) != 24:
                return [ValidationResultModel(
                    level=ValidationLevel.ERROR,
                    code=ValidationCode.MISSING_REQUIRED_DATA,
                    message="Incomplete diurnal data: need 24 hourly values",
                    layer="L3",
                    scope="diurnal_pattern_validation"
                )]
            
            validation_issues = []
            
            # Validate GPP diurnal pattern
            gpp_max_hour = np.argmax(gpp)
            gpp_night_avg = np.mean([gpp[i] for i in range(24) if hours[i] < 6 or hours[i] > 20])
            gpp_day_avg = np.mean([gpp[i] for i in range(24) if 10 <= hours[i] <= 16])
            
            # GPP should peak around midday (10-14h)
            if not (10 <= gpp_max_hour <= 14):
                validation_issues.append(f"GPP peaks at hour {gpp_max_hour}, expected 10-14")
            
            # Nighttime GPP should be near zero
            if gpp_night_avg > 2.0:  # 渭mol CO2 m-2 s-1
                validation_issues.append(f"Nighttime GPP {gpp_night_avg:.1f} too high (expected <2)")
            
            # Day/night ratio should be substantial
            if gpp_night_avg > 0:
                day_night_ratio = gpp_day_avg / gpp_night_avg
                if day_night_ratio < 5:
                    validation_issues.append(f"GPP day/night ratio {day_night_ratio:.1f} too low (expected >5)")
            
            # Validate ET diurnal pattern
            et_max_hour = np.argmax(et)
            et_night_avg = np.mean([et[i] for i in range(24) if hours[i] < 6 or hours[i] > 20])
            et_day_avg = np.mean([et[i] for i in range(24) if 10 <= hours[i] <= 16])
            
            # ET should peak in afternoon (12-16h)
            if not (12 <= et_max_hour <= 16):
                validation_issues.append(f"ET peaks at hour {et_max_hour}, expected 12-16")
            
            # Validate light response
            if len(par) == 24:
                # Calculate correlation between PAR and GPP during daylight hours
                daylight_mask = np.array(par) > 50  # 渭mol m-2 s-1
                if np.sum(daylight_mask) > 3:  # Need at least 4 daylight hours
                    gpp_day = np.array(gpp)[daylight_mask]
                    par_day = np.array(par)[daylight_mask]
                    
                    if np.var(par_day) > 1e-10 and np.var(gpp_day) > 1e-10:
                        # Manual correlation calculation to avoid corrcoef issues
                        try:
                            par_mean = np.mean(par_day)
                            gpp_mean = np.mean(gpp_day)
                            
                            par_centered = par_day - par_mean
                            gpp_centered = gpp_day - gpp_mean
                            
                            numerator = np.sum(par_centered * gpp_centered)
                            par_ss = np.sum(par_centered ** 2)
                            gpp_ss = np.sum(gpp_centered ** 2)
                            
                            if par_ss > 1e-10 and gpp_ss > 1e-10:
                                par_gpp_correlation = numerator / np.sqrt(par_ss * gpp_ss)
                                par_gpp_correlation = np.clip(par_gpp_correlation, -1.0, 1.0)
                                if not np.isfinite(par_gpp_correlation):
                                    par_gpp_correlation = 0.0
                            else:
                                par_gpp_correlation = 0.0
                        except Exception:
                            par_gpp_correlation = 0.0
                        if par_gpp_correlation < 0.6:
                            validation_issues.append(f"PAR-GPP correlation {par_gpp_correlation:.2f} too low (expected >0.6)")
            
            # Determine validation result
            if not validation_issues:
                level = ValidationLevel.INFO
                message = f"Diurnal patterns realistic: GPP peak at {gpp_max_hour}h, ET peak at {et_max_hour}h"
            elif len(validation_issues) <= 2:
                level = ValidationLevel.WARNING
                message = f"Diurnal pattern issues: {'; '.join(validation_issues[:2])}"
            else:
                level = ValidationLevel.ERROR
                message = f"Diurnal patterns unrealistic: {'; '.join(validation_issues[:3])}"
            
            result = ValidationResultModel(
                level=level,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,  # TODO: Add DIURNAL_PATTERN codes
                message=message,
                layer="L3",
                scope="diurnal_pattern_validation",
                tags=["diurnal", "gpp", "et", "light_response"],
                metrics={
                    "gpp_max_hour": int(gpp_max_hour),
                    "et_max_hour": int(et_max_hour),
                    "gpp_day_avg": float(gpp_day_avg),
                    "gpp_night_avg": float(gpp_night_avg),
                    "et_day_avg": float(et_day_avg),
                    "et_night_avg": float(et_night_avg),
                    "validation_issues_count": len(validation_issues)
                }
            )
            
            # Add diurnal pattern detail
            result.add_detail(
                variable_name="diurnal_patterns",
                units="mixed",
                value=gpp_max_hour,
                statistics={
                    "hourly_gpp": list(gpp),
                    "hourly_et": list(et),
                    "hourly_par": list(par) if len(par) == 24 else [],
                    "gpp_amplitude": float(max(gpp) - min(gpp)),
                    "et_amplitude": float(max(et) - min(et)),
                    "validation_issues": validation_issues
                }
            )
            
            results.append(result)
            return results
        
        # Create realistic diurnal data
        hours = list(range(24))
        
        # Realistic diurnal GPP pattern
        gpp_diurnal = [
            0, 0, 0, 0, 0, 1, 5, 12, 18, 22, 25, 26, 
            24, 22, 18, 14, 8, 4, 1, 0, 0, 0, 0, 0
        ]
        
        # Realistic diurnal ET pattern (peaks later than GPP)
        et_diurnal = [
            0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2, 0.4, 
            0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.4, 1.2, 
            0.9, 0.6, 0.3, 0.1, 0.05, 0.05, 0.05, 0.05
        ]
        
        # Realistic PAR pattern
        par_diurnal = [
            0, 0, 0, 0, 0, 50, 200, 600, 1000, 1400, 1700, 1800,
            1700, 1500, 1200, 800, 400, 100, 20, 0, 0, 0, 0, 0
        ]
        
        diurnal_scenario = {
            'diurnal_data': {
                'hour': hours,
                'gpp': gpp_diurnal,
                'et': et_diurnal,
                'par': par_diurnal
            }
        }
        
        test_context = ValidationContext(config=diurnal_scenario)
        results = validate_diurnal_patterns(test_context)
        
        assert len(results) == 1
        result = results[0]
        
        # Should validate as realistic diurnal patterns
        assert result.level == ValidationLevel.INFO
        assert 10 <= result.metrics['gpp_max_hour'] <= 14  # GPP peaks around midday
        assert 12 <= result.metrics['et_max_hour'] <= 16  # ET peaks in afternoon
        assert result.metrics['gpp_night_avg'] < 2.0  # Low nighttime GPP
        
        # Check diurnal detail
        detail = result.details[0]
        assert len(detail.statistics['hourly_gpp']) == 24
        assert len(detail.statistics['validation_issues']) == 0


class TestProductionSystemReadiness:
    """Test production system readiness and operational validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ValidationEngine()
    
    def test_operational_readiness_validation(self):
        """Test operational readiness for production deployment."""
        @validated_rule(
            rule_id="operational_readiness",
            layer="L3",
            domains=["system"],
            tags=["production", "readiness", "operational"]
        )
        def validate_operational_readiness(context):
            """Validate system readiness for operational deployment."""
            results = []
            
            system_config = context.config.get('system_config', {})
            performance_data = context.config.get('performance_data', {})
            
            readiness_issues = []
            readiness_score = 0
            max_score = 10
            
            # 1. Performance Requirements (2 points)
            execution_time = performance_data.get('execution_time', float('inf'))
            memory_usage = performance_data.get('memory_usage_mb', float('inf'))
            
            if execution_time < 300:  # 5 minutes
                readiness_score += 1
            if memory_usage < 2000:  # 2GB
                readiness_score += 1
            elif execution_time >= 300:
                readiness_issues.append(f"Execution time {execution_time}s exceeds 300s limit")
            elif memory_usage >= 2000:
                readiness_issues.append(f"Memory usage {memory_usage}MB exceeds 2GB limit")
            
            # 2. Data Quality Requirements (2 points)
            data_completeness = system_config.get('data_completeness', 0.0)
            data_quality_score = system_config.get('data_quality_score', 0.0)
            
            if data_completeness > 0.95:
                readiness_score += 1
            else:
                readiness_issues.append(f"Data completeness {data_completeness:.1%} below 95% requirement")
                
            if data_quality_score > 0.8:
                readiness_score += 1
            else:
                readiness_issues.append(f"Data quality score {data_quality_score:.2f} below 0.8 requirement")
            
            # 3. Validation Coverage (2 points)
            l1_pass_rate = system_config.get('l1_pass_rate', 0.0)
            l2_pass_rate = system_config.get('l2_pass_rate', 0.0)
            
            if l1_pass_rate >= 0.95:
                readiness_score += 1
            else:
                readiness_issues.append(f"L1 validation pass rate {l1_pass_rate:.1%} below 95%")
                
            if l2_pass_rate >= 0.90:
                readiness_score += 1
            else:
                readiness_issues.append(f"L2 validation pass rate {l2_pass_rate:.1%} below 90%")
            
            # 4. Scientific Accuracy (2 points)
            benchmark_correlation = system_config.get('benchmark_correlation', 0.0)
            conservation_residuals = system_config.get('max_conservation_residual', float('inf'))
            
            if benchmark_correlation > 0.7:
                readiness_score += 1
            else:
                readiness_issues.append(f"Benchmark correlation {benchmark_correlation:.2f} below 0.7")
                
            if conservation_residuals < 1e-4:
                readiness_score += 1
            else:
                readiness_issues.append(f"Conservation residuals {conservation_residuals:.2e} above 1e-4")
            
            # 5. System Reliability (2 points)
            error_rate = system_config.get('system_error_rate', 1.0)
            uptime = system_config.get('system_uptime', 0.0)
            
            if error_rate < 0.01:  # <1% error rate
                readiness_score += 1
            else:
                readiness_issues.append(f"System error rate {error_rate:.1%} above 1%")
                
            if uptime > 0.99:  # >99% uptime
                readiness_score += 1
            else:
                readiness_issues.append(f"System uptime {uptime:.1%} below 99%")
            
            # Determine readiness level
            readiness_percentage = (readiness_score / max_score) * 100
            
            if readiness_score >= 9:
                level = ValidationLevel.INFO
                message = f"System production ready: {readiness_percentage:.0f}% readiness score"
            elif readiness_score >= 7:
                level = ValidationLevel.WARNING
                message = f"System near production ready: {readiness_percentage:.0f}% readiness, {len(readiness_issues)} issues"
            else:
                level = ValidationLevel.ERROR
                message = f"System not production ready: {readiness_percentage:.0f}% readiness, {len(readiness_issues)} issues"
            
            result = ValidationResultModel(
                level=level,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,  # TODO: Add OPERATIONAL_READINESS codes
                message=message,
                layer="L3",
                scope="operational_readiness",
                tags=["production", "readiness", "operational"],
                metrics={
                    "readiness_score": readiness_score,
                    "readiness_percentage": readiness_percentage,
                    "readiness_issues_count": len(readiness_issues),
                    "max_possible_score": max_score
                }
            )
            
            # Add readiness assessment detail
            result.add_detail(
                variable_name="operational_readiness_assessment",
                units="score",
                value=readiness_score,
                threshold=7,  # Minimum for production
                statistics={
                    "performance_score": 2 if execution_time < 300 and memory_usage < 2000 else 1,
                    "data_quality_score": int(data_completeness > 0.95) + int(data_quality_score > 0.8),
                    "validation_coverage_score": int(l1_pass_rate >= 0.95) + int(l2_pass_rate >= 0.90),
                    "scientific_accuracy_score": int(benchmark_correlation > 0.7) + int(conservation_residuals < 1e-4),
                    "system_reliability_score": int(error_rate < 0.01) + int(uptime > 0.99),
                    "readiness_issues": readiness_issues,
                    "readiness_criteria": {
                        "execution_time_limit": 300,
                        "memory_limit_mb": 2000,
                        "min_data_completeness": 0.95,
                        "min_benchmark_correlation": 0.7,
                        "max_error_rate": 0.01
                    }
                }
            )
            
            results.append(result)
            return results
        
        # Test production-ready system configuration
        production_ready_config = {
            'system_config': {
                'data_completeness': 0.98,
                'data_quality_score': 0.85,
                'l1_pass_rate': 0.97,
                'l2_pass_rate': 0.92,
                'benchmark_correlation': 0.78,
                'max_conservation_residual': 5e-5,
                'system_error_rate': 0.005,
                'system_uptime': 0.995
            },
            'performance_data': {
                'execution_time': 180,  # 3 minutes
                'memory_usage_mb': 1200  # 1.2 GB
            }
        }
        
        test_context = ValidationContext(config=production_ready_config)
        results = validate_operational_readiness(test_context)
        
        assert len(results) == 1
        result = results[0]
        
        # Should validate as production ready
        assert result.level == ValidationLevel.INFO
        assert result.metrics['readiness_score'] >= 9
        assert result.metrics['readiness_percentage'] >= 90.0
        
        # Check readiness detail
        detail = result.details[0]
        assert detail.variable_name == "operational_readiness_assessment"
        assert detail.value >= 7  # Above production threshold
        assert len(detail.statistics['readiness_issues']) == 0  # No blocking issues


if __name__ == "__main__":
    pytest.main([__file__])


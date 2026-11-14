Core Scientific Class Reference
===============================

The scientific reference covers 94 certificate-backed classes: 90 executable concrete Functors, 4 fail-closed unavailable boundaries, and 0 abstract scientific contracts. The class-to-certificate relationship is one-to-one.

Every executable concrete process follows the current ``abstraction -> bridge -> FunctorBase -> ParameterManager -> StateManager`` lifecycle. The pages below report unavailable classes separately; their constructors are not invoked. The pages document exact signatures, variable contracts, governing equations, parameters, units, structural roles, ports, and references.

.. toctree::
   :maxdepth: 2
   :caption: Scientific domains

   anthroposphere
   atmosphere
   biosphere
   hydrosphere
   pedosphere
   isotopes
   transport

Complete coverage matrix
------------------------

.. list-table::
   :header-rows: 1
   :widths: 12 24 20 14 14 10 20

   * - Domain
     - Class
     - Bridge
     - Role
     - Spatial scale
     - IPHS level
     - Implementation status
   * - anthroposphere
     - :ref:`atmosphere_coupling <class-relic-scientific-anthroposphere-emissions-coupling-to-atmosphere-atmospherecoupling>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``runtime_unbound``
     - ``level_0``
     - ``not_implemented_fail_closed``
   * - anthroposphere
     - :ref:`sector_allocation <class-relic-scientific-anthroposphere-emissions-sector-allocation-sectorallocation>`
     - ``ScaleCouplingFunctor``
     - ``scale_coupling``
     - ``grid_cell``
     - ``level_1``
     - ``implemented``
   * - anthroposphere
     - :ref:`species_mapping <class-relic-scientific-anthroposphere-emissions-species-mapping-speciesmapper>`
     - ``ScaleCouplingFunctor``
     - ``scale_coupling``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - anthroposphere
     - :ref:`anthroposphere_interface_mapping <class-relic-scientific-anthroposphere-interface-mapping-anthroposphereinterfacemapping>`
     - ``ScaleCouplingFunctor``
     - ``scale_coupling``
     - ``runtime_unbound``
     - ``level_0``
     - ``not_implemented_fail_closed``
   * - anthroposphere
     - :ref:`emission_inventory <class-relic-scientific-anthroposphere-inventory-emissioninventory>`
     - ``ScaleCouplingFunctor``
     - ``scale_coupling``
     - ``runtime_unbound``
     - ``level_0``
     - ``not_implemented_fail_closed``
   * - anthroposphere
     - :ref:`land_use_parameter_mapping <class-relic-scientific-anthroposphere-land-use-mapping-landuseparametermapping>`
     - ``PhysicalFieldFunctor``
     - ``observation``
     - ``grid_cell``
     - ``level_1``
     - ``implemented``
   * - anthroposphere
     - :ref:`land_use_transitions <class-relic-scientific-anthroposphere-land-use-transitions-landusetransitions>`
     - ``ScaleCouplingFunctor``
     - ``scale_coupling``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - anthroposphere
     - :ref:`management_coordinator <class-relic-scientific-anthroposphere-management-coordinator-managementcoordinator>`
     - ``ScaleCouplingFunctor``
     - ``scale_coupling``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - anthroposphere
     - :ref:`fertilization_management <class-relic-scientific-anthroposphere-management-fertilization-fertilizationmanagement>`
     - ``TransportMechanismFunctor``
     - ``port``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - anthroposphere
     - :ref:`fire_management <class-relic-scientific-anthroposphere-management-fire-management-firemanagement>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - anthroposphere
     - :ref:`grazing_management <class-relic-scientific-anthroposphere-management-grazing-grazingmanagement>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - anthroposphere
     - :ref:`harvest_management <class-relic-scientific-anthroposphere-management-harvest-harvestmanagement>`
     - ``BiologicalProcessFunctor``
     - ``port``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - anthroposphere
     - :ref:`IrrigationManagement <class-relic-scientific-anthroposphere-management-irrigation-irrigationmanagement>`
     - ``TransportMechanismFunctor``
     - ``port``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - anthroposphere
     - :ref:`TillageManagement <class-relic-scientific-anthroposphere-management-tillage-tillagemanagement>`
     - ``TransportMechanismFunctor``
     - ``port``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - atmosphere
     - :ref:`ChemistryFunctor <class-relic-scientific-atmosphere-chemistry-chemistryfunctor>`
     - ``PhysicalFieldFunctor``
     - ``field``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - atmosphere
     - :ref:`ChemistryFunctor <class-atmosphere-chemistry-chemistry-functor-closed-constant-volume>`
     - ``PhysicalFieldFunctor``
     - ``field``
     - ``grid_cell``
     - ``level_5``
     - ``implemented``
   * - atmosphere
     - :ref:`EnvironmentalForcingFunctor <class-relic-scientific-atmosphere-environmental-forcing-environmentalforcingfunctor>`
     - ``PhysicalFieldFunctor``
     - ``port``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - atmosphere
     - :ref:`RadiationEnergyConservationValidator <class-relic-scientific-atmosphere-radiation-energy-conservation-radiationenergyconservationvalidator>`
     - ``PhysicalFieldFunctor``
     - ``observation``
     - ``point``
     - ``level_1``
     - ``implemented``
   * - atmosphere
     - :ref:`RadiationField <class-relic-scientific-atmosphere-radiation-radiationfield>`
     - ``PhysicalFieldFunctor``
     - ``field``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - atmosphere
     - :ref:`RadiationFunctor <class-relic-scientific-atmosphere-radiation-radiationfunctor>`
     - ``PhysicalFieldFunctor``
     - ``field``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - atmosphere
     - :ref:`MarkedPoissonRainfallFunctor <class-relic-scientific-atmosphere-stochastic-rainfall-markedpoissonrainfallfunctor>`
     - ``PhysicalFieldFunctor``
     - ``field``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - atmosphere
     - :ref:`OrnsteinUhlenbeckTemperatureFunctor <class-relic-scientific-atmosphere-stochastic-temperature-ornsteinuhlenbecktemperaturefunctor>`
     - ``PhysicalFieldFunctor``
     - ``field``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - atmosphere
     - :ref:`AtmosphericAdvectionDiffusionFunctor <class-relic-scientific-atmosphere-transport-atmosphericadvectiondiffusionfunctor-atmosphericadvectiondiffusionfunctor>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - atmosphere
     - :ref:`BoundaryLayerEmissionSource <class-relic-scientific-atmosphere-transport-boundary-layer-emission-source-boundarylayeremissionsource>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``grid_cell``
     - ``level_1``
     - ``implemented``
   * - atmosphere
     - :ref:`DryDepositionAggregator <class-relic-scientific-atmosphere-transport-dry-deposition-aggregator-drydepositionaggregator>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - atmosphere
     - :ref:`EddyDiffusionBLMFunctor <class-relic-scientific-atmosphere-transport-eddy-diffusion-blm-eddydiffusionblmfunctor>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - atmosphere
     - :ref:`TurbulenceFunctor <class-relic-scientific-atmosphere-turbulence-turbulencefunctor>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - biosphere
     - :ref:`MEGANModel <class-relic-scientific-biosphere-bvoc-megan-model-meganmodel>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - biosphere
     - :ref:`IsopreneSynthesis <class-relic-scientific-biosphere-bvoc-synthesis-isoprene-isoprenesynthesis>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - biosphere
     - :ref:`MonoterpeneSynthesis <class-relic-scientific-biosphere-bvoc-synthesis-monoterpene-monoterpenesynthesis>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - biosphere
     - :ref:`SesquiterpeneStressModel <class-relic-scientific-biosphere-bvoc-synthesis-sesquiterpene-sesquiterpenestressmodel>`
     - ``BiologicalProcessFunctor``
     - ``observation``
     - ``grid_cell``
     - ``level_1``
     - ``implemented``
   * - biosphere
     - :ref:`SesquiterpeneSynthesis <class-relic-scientific-biosphere-bvoc-synthesis-sesquiterpene-sesquiterpenesynthesis>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - biosphere
     - :ref:`BiomassAllocation <class-relic-scientific-biosphere-growth-biomass-allocation-biomassallocation>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - biosphere
     - :ref:`PhenologyGrowth <class-relic-scientific-biosphere-growth-phenology-growth-phenologygrowth>`
     - ``BiologicalProcessFunctor``
     - ``observation``
     - ``patch``
     - ``level_1``
     - ``implemented``
   * - biosphere
     - :ref:`StructuralGrowth <class-relic-scientific-biosphere-growth-structural-growth-structuralgrowth>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``patch``
     - ``level_2``
     - ``implemented``
   * - biosphere
     - :ref:`CropPhenology <class-relic-scientific-biosphere-phenology-models-cropphenology>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - biosphere
     - :ref:`GDDPhenology <class-relic-scientific-biosphere-phenology-models-gddphenology>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - biosphere
     - :ref:`GSIPhenology <class-relic-scientific-biosphere-phenology-models-gsiphenology>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - biosphere
     - :ref:`StaticPhenology <class-relic-scientific-biosphere-phenology-models-staticphenology>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - biosphere
     - :ref:`PhenologyFunctor <class-relic-scientific-biosphere-phenology-phenology-functor-phenologyfunctor>`
     - ``BiologicalProcessFunctor``
     - ``observation``
     - ``grid_cell``
     - ``level_1``
     - ``implemented``
   * - biosphere
     - :ref:`FarquharPhotosynthesis <class-relic-scientific-biosphere-photosynthesis-farquhar-farquharphotosynthesis>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``patch``
     - ``level_2``
     - ``implemented``
   * - biosphere
     - :ref:`LeafEnergyBalance <class-relic-scientific-biosphere-photosynthesis-leaf-energy-balance-leafenergybalance>`
     - ``BiologicalProcessFunctor``
     - ``observation``
     - ``patch``
     - ``level_1``
     - ``implemented``
   * - biosphere
     - :ref:`MedlynStomatalConductance <class-relic-scientific-biosphere-photosynthesis-stomatal-medlynstomatalconductance>`
     - ``BiologicalProcessFunctor``
     - ``observation``
     - ``patch``
     - ``level_1``
     - ``implemented``
   * - biosphere
     - :ref:`AutotrophicRespiration <class-relic-scientific-biosphere-respiration-autotrophic-respiration-autotrophicrespiration>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - biosphere
     - :ref:`GrowthRespiration <class-relic-scientific-biosphere-respiration-growth-respiration-growthrespiration>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - biosphere
     - :ref:`MaintenanceRespiration <class-relic-scientific-biosphere-respiration-maintenance-respiration-maintenancerespiration>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - biosphere
     - :ref:`NutrientStress <class-relic-scientific-biosphere-stress-nutrient-stress-nutrientstress>`
     - ``BiologicalProcessFunctor``
     - ``observation``
     - ``patch``
     - ``level_1``
     - ``implemented``
   * - biosphere
     - :ref:`OzoneStress <class-relic-scientific-biosphere-stress-ozone-stress-ozonestress>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``patch``
     - ``level_2``
     - ``implemented``
   * - biosphere
     - :ref:`ThermalStress <class-relic-scientific-biosphere-stress-thermal-stress-thermalstress>`
     - ``BiologicalProcessFunctor``
     - ``observation``
     - ``patch``
     - ``level_1``
     - ``implemented``
   * - biosphere
     - :ref:`WaterStress <class-relic-scientific-biosphere-stress-water-stress-waterstress>`
     - ``BiologicalProcessFunctor``
     - ``observation``
     - ``patch``
     - ``level_1``
     - ``implemented``
   * - biosphere
     - :ref:`SimpleCanopyPhotosynthesis <class-relic-scientific-biosphere-surrogates-simple-photosynthesis-simplecanopyphotosynthesis>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - hydrosphere
     - :ref:`penman_monteith_et <class-relic-scientific-hydrosphere-evapotranspiration-penman-monteith-penmanmonteithet>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - hydrosphere
     - :ref:`priestley_taylor_et <class-relic-scientific-hydrosphere-evapotranspiration-priestley-taylor-priestleytayloret>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - hydrosphere
     - :ref:`groundwater_dynamics <class-relic-scientific-hydrosphere-groundwater-dynamics-groundwaterdynamics>`
     - ``PhysicalFieldFunctor``
     - ``field``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - hydrosphere
     - :ref:`subsurface_runoff <class-relic-scientific-hydrosphere-runoff-subsurface-subsurfacerunoff>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``patch``
     - ``level_3``
     - ``implemented``
   * - hydrosphere
     - :ref:`HortonianPondingEventFunctor <class-relic-scientific-hydrosphere-runoff-surface-ponding-hortonianpondingeventfunctor>`
     - ``PhysicalFieldFunctor``
     - ``field``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - hydrosphere
     - :ref:`SurfacePondingFunctor <class-relic-scientific-hydrosphere-runoff-surface-ponding-surfacepondingfunctor>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - hydrosphere
     - :ref:`surface_runoff <class-relic-scientific-hydrosphere-runoff-surface-surfacerunoff>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - hydrosphere
     - :ref:`snow_albedo <class-relic-scientific-hydrosphere-snow-albedo-snowalbedo>`
     - ``PhysicalFieldFunctor``
     - ``observation``
     - ``grid_cell``
     - ``level_1``
     - ``implemented``
   * - hydrosphere
     - :ref:`snow_pack <class-relic-scientific-hydrosphere-snow-snowpack-snowpack>`
     - ``PhysicalFieldFunctor``
     - ``field``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - hydrosphere
     - :ref:`bucket_soil_moisture <class-relic-scientific-hydrosphere-soil-moisture-bucket-bucketsoilmoisture>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - hydrosphere
     - :ref:`SoilWaterRetention <class-relic-scientific-hydrosphere-soil-moisture-retention-soilwaterretention>`
     - ``PhysicalFieldFunctor``
     - ``observation``
     - ``grid_cell``
     - ``level_1``
     - ``implemented``
   * - hydrosphere
     - :ref:`richards_soil_moisture <class-relic-scientific-hydrosphere-soil-moisture-richards-richardssoilmoisture>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - hydrosphere
     - :ref:`capillary_rise_transport <class-relic-scientific-hydrosphere-transport-capillary-rise-capillaryrisetransport>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - hydrosphere
     - :ref:`vertical_percolation_transport <class-relic-scientific-hydrosphere-transport-vertical-percolation-verticalpercolationtransport>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - pedosphere
     - :ref:`CarbonPoolsFunctor <class-relic-scientific-pedosphere-carbon-pools-carbon-pools-functor-carbonpoolsfunctor>`
     - ``BiologicalProcessFunctor``
     - ``observation``
     - ``grid_cell``
     - ``level_1``
     - ``implemented``
   * - pedosphere
     - :ref:`PoolsMatrix <class-relic-scientific-pedosphere-carbon-pools-pools-matrix-poolsmatrix>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - pedosphere
     - :ref:`FirstOrderDecomposition <class-relic-scientific-pedosphere-decomposition-first-order-firstorderdecomposition>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - pedosphere
     - :ref:`MicrobialEnzymeDecomposition <class-relic-scientific-pedosphere-decomposition-microbial-enzyme-microbialenzymedecomposition>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - pedosphere
     - :ref:`LitterfallInterfaceMapper <class-relic-scientific-pedosphere-interface-mapping-litterfallinterfacemapper>`
     - ``ScaleCouplingFunctor``
     - ``scale_coupling``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - pedosphere
     - :ref:`PedosphereInterfaceMapper <class-relic-scientific-pedosphere-interface-mapping-pedosphereinterfacemapper>`
     - ``ScaleCouplingFunctor``
     - ``scale_coupling``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - pedosphere
     - :ref:`MethaneBudgetCoupler <class-relic-scientific-pedosphere-methane-budget-coupler-methanebudgetcoupler>`
     - ``ScaleCouplingFunctor``
     - ``scale_coupling``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - pedosphere
     - :ref:`MethaneOxidationFunctor <class-relic-scientific-pedosphere-methane-oxidation-methaneoxidationfunctor>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - pedosphere
     - :ref:`MethaneProductionFunctor <class-relic-scientific-pedosphere-methane-production-methaneproductionfunctor>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - pedosphere
     - :ref:`MethaneScaleCouplingFunctor <class-relic-scientific-pedosphere-methane-scales-methanescalecouplingfunctor>`
     - ``ScaleCouplingFunctor``
     - ``observation``
     - ``grid_cell``
     - ``level_1``
     - ``implemented``
   * - pedosphere
     - :ref:`MethaneTransportFunctor <class-relic-scientific-pedosphere-methane-transport-methanetransportfunctor>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - pedosphere
     - :ref:`denitrification_process <class-relic-scientific-pedosphere-nutrient-cycling-denitrification-denitrificationprocess>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - pedosphere
     - :ref:`MineralizationProcess <class-relic-scientific-pedosphere-nutrient-cycling-mineralization-mineralizationprocess>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - pedosphere
     - :ref:`nitrification_process <class-relic-scientific-pedosphere-nutrient-cycling-nitrification-nitrificationprocess>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - pedosphere
     - :ref:`NitrogenLeachingProcess <class-relic-scientific-pedosphere-nutrient-cycling-nitrogen-leaching-nitrogenleachingprocess>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - pedosphere
     - :ref:`NitrogenCycling <class-relic-scientific-pedosphere-nutrient-cycling-nitrogen-nitrogencycling>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - pedosphere
     - :ref:`RootNutrientUptake <class-relic-scientific-pedosphere-nutrient-cycling-root-nutrient-uptake-rootnutrientuptake>`
     - ``BiologicalProcessFunctor``
     - ``reaction``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - pedosphere
     - :ref:`StoichiometryConstraint <class-relic-scientific-pedosphere-nutrient-cycling-stoichiometry-stoichiometryconstraint>`
     - ``ScaleCouplingFunctor``
     - ``observation``
     - ``grid_cell``
     - ``level_1``
     - ``implemented``
   * - pedosphere
     - :ref:`EquilibriumSorptionFunctor <class-relic-scientific-pedosphere-transport-equilibrium-sorption-equilibriumsorptionfunctor>`
     - ``PhysicalFieldFunctor``
     - ``field``
     - ``grid_cell``
     - ``level_2``
     - ``implemented``
   * - pedosphere
     - :ref:`GasDiffusion1D <class-relic-scientific-pedosphere-transport-gas-diffusion-gasdiffusion1d>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - pedosphere
     - :ref:`MovingDomainDiffusionFunctor <class-relic-scientific-pedosphere-transport-moving-domain-diffusion-movingdomaindiffusionfunctor>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - pedosphere
     - :ref:`soil_water_transport <class-relic-scientific-pedosphere-transport-soilwatertransportfunctor-soilwatertransportfunctor>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - pedosphere
     - :ref:`SoluteAdvectionDiffusion1D <class-relic-scientific-pedosphere-transport-solute-advection-diffusion-soluteadvectiondiffusion1d>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - pedosphere
     - :ref:`TwoRegionMassTransferFunctor <class-relic-scientific-pedosphere-transport-two-region-mass-transfer-tworegionmasstransferfunctor>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - isotopes
     - :ref:`IsotopeHeartbeatFunctor <class-relic-scientific-isotopes-functor-isotopeheartbeatfunctor>`
     - ``ScaleCouplingFunctor``
     - ``scale_coupling``
     - ``runtime_unbound``
     - ``level_0``
     - ``not_implemented_fail_closed``
   * - transport
     - :ref:`advection_diffusion <class-relic-scientific-transport-advection-diffusion-advectiondiffusion>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - transport
     - :ref:`linear_advection <class-relic-scientific-transport-advection-linearadvection>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - transport
     - :ref:`fickian_diffusion <class-relic-scientific-transport-diffusion-fickiandiffusion>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``grid_cell``
     - ``level_3``
     - ``implemented``
   * - transport
     - :ref:`fickian_diffusion <class-transport-diffusion-fickian-diffusion-strict-runtime-point>`
     - ``TransportMechanismFunctor``
     - ``transport``
     - ``grid_cell``
     - ``level_4``
     - ``implemented``

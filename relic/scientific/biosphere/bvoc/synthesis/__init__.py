"""
BVOC Synthesis Pathways for RELIC

This module provides detailed biochemical synthesis pathway models for specific
BVOC compounds. These models complement the MEGAN framework by providing
mechanistic understanding of compound-specific biosynthesis processes.

Main Components
---------------
- IsopreneSynthesis: MEP pathway-based isoprene biosynthesis model
- MonoterpeneSynthesis: Monoterpene biosynthesis from GPP and IPP pools  
- SesquiterpeneSynthesis: Sesquiterpene synthesis pathway model

Biochemical Pathways
--------------------
1. Isoprene: MEP (methylerythritol phosphate) pathway
   - Direct synthesis from recent photosynthates
   - Light and temperature dependent enzyme activities
   - Immediate emission upon synthesis

2. Monoterpenes: Monoterpene synthase pathway
   - Synthesis from stored isoprenyl diphosphate pools
   - Temperature dependent but light independent
   - Storage in resin ducts and specialized structures

3. Sesquiterpenes: Farnesyl diphosphate pathway  
   - Complex terpene synthesis from C15 precursors
   - High temperature sensitivity
   - Often induced by stress or damage

Integration with MEGAN
----------------------
These mechanistic models provide:
- Biochemically-based parameter estimation
- Process-level understanding for parameter sensitivity
- Alternative emission calculation approaches
- Validation for empirical MEGAN relationships

References
----------
- Lichtenthaler, H.K. (1999). The 1-deoxy-D-xylulose-5-phosphate pathway of
  isoprenoid biosynthesis in plants.
  Annual Review of Plant Physiology and Plant Molecular Biology 50, 47-65
- Dudareva, N., et al. (2006). Plant volatiles: recent advances and future
  perspectives. Critical Reviews in Plant Sciences 25, 417-440
- Vickers, C.E., et al. (2009). A unified mechanism of action for volatile 
  isoprenoids in plant abiotic stress. Nature Chemical Biology 5, 283-291
"""

from .isoprene import IsopreneSynthesis, MEPPathwayModel
from .monoterpene import MonoterpeneSynthesis, MonoterpenePoolModel
# Parameter management imports

__all__ = [
    # Isoprene synthesis
    'IsopreneSynthesis',
    'MEPPathwayModel',
    
    # Monoterpene synthesis  
    'MonoterpeneSynthesis',
    'MonoterpenePoolModel',
]

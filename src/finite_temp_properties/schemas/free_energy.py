"""What the analysis jobs return. Energies eV/atom, temperatures K, P = 0."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SolidFreeEnergyDoc(BaseModel):
    """Crystal free energy at one temperature, from a Frenkel-Ladd switch."""

    temperature: float = Field(description="anchor temperature T0 (K)")
    species: list[str] = Field(description="elements in LAMMPS type order")
    free_energy: float = Field(description="F(T0) = F_einstein - W, eV/atom")
    einstein_reference: float = Field(description="Einstein crystal F, eV/atom")
    work: float = Field(description="mean switching work W, eV/atom")
    hysteresis: float = Field(description="|W_fwd - W_bwd|, the error scale")
    spring_constants: list[float] = Field(description="k_i = 3kBT/<dr^2>_i, eV/A^2")
    volume_per_atom: float = Field(description="A^3/atom at the anchor")
    natoms: int
    msd_dir: Optional[str] = None
    frenkel_ladd_dir: Optional[str] = None


class LiquidFreeEnergyDoc(BaseModel):
    """Melt free energy at the anchor, from the two-leg Uhlenbeck-Ford path."""

    temperature: float = Field(description="melt anchor Ta (K)")
    species: list[str] = Field(description="elements in LAMMPS type order")
    free_energy: float = Field(
        description="F(Ta) = F_UF + F_ideal_gas - W_leg2 - W_leg1, eV/atom")
    reference_free_energy: float = Field(description="F_UF + F_ideal_gas, eV/atom")
    work_leg1: float = Field(description="real -> multi-sigma UFM, eV/atom")
    work_leg2: float = Field(description="multi-sigma -> single-sigma UFM, eV/atom")
    hysteresis: float = Field(description="summed |W_fwd - W_bwd| of both legs")
    ufm_p: float = Field(description="UF coupling p (eps = p kB T)")
    sigmas: list[float] = Field(description="per-pair sigma from the partial RDFs, A")
    sigma_single: float = Field(description="single-fluid sigma placing x on X_TARGET, A")
    enthalpy_per_atom: float = Field(description="H(Ta) of the reference melt, eV/atom")
    volume_per_atom: float
    natoms: int
    melt_dir: Optional[str] = None
    leg1_dir: Optional[str] = None
    leg2_dir: Optional[str] = None


class PotentialSwitchDoc(BaseModel):
    """dG of transporting one phase from the reference to the target potential."""

    temperature: float
    dG: float = Field(description="dG(reference -> target), eV/atom")
    hysteresis: float
    volume_reference: float = Field(description="lambda=0 volume, A^3/atom")
    volume_target: float = Field(
        description="lambda=1 volume, A^3/atom -- must match an independent "
                    "target-potential NPT of this phase")
    dir_name: Optional[str] = None


class GibbsCurveDoc(BaseModel):
    """G(T) of the crystal and the amorphous phase on a common grid."""

    temperatures: list[float]
    g_crystal: Optional[list[float]] = Field(None, description="eV/atom")
    g_amorphous: Optional[list[float]] = Field(None, description="eV/atom")

    t_crystal_anchor: Optional[float] = None
    h_crystal_anchor: Optional[float] = Field(None, description="H(T0), eV/atom")
    g_crystal_anchor: Optional[float] = Field(
        None, description="G(T0) = F_TI + dG_switch, eV/atom")
    s_crystal_anchor: Optional[float] = Field(None, description="S(T0), J/mol-atom/K")

    t_melt_anchor: Optional[float] = None
    g_melt_anchor: Optional[float] = None
    s_melt_anchor: Optional[float] = Field(None, description="S(Ta), J/mol-atom/K")

    solid: Optional[SolidFreeEnergyDoc] = None
    liquid: Optional[LiquidFreeEnergyDoc] = None
    solid_switch: Optional[PotentialSwitchDoc] = None
    liquid_switch: Optional[PotentialSwitchDoc] = None

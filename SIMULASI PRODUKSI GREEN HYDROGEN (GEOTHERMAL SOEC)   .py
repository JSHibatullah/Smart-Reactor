import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass(frozen=True)
class SystemConstants:
    """Menyimpan parameter tetapan fisika dan batasan operasional."""
    MW_H2: float = 2.016      # Berat molekul H2 (kg/kmol)
    MW_H2O: float = 18.015    # Berat molekul H2O (kg/kmol)
    EFFICIENCY_CONV: float = 0.85  # Efisiensi konversi massa H2O -> H2
    
    TARGET_TEMP_SOEC: float = 800.0  # Celsius
    TARGET_ENTHALPY: float = 4100.0  # kJ/kg (Entalpi uap di ~800C)
    TARGET_PRESSURE: float = 20.0    # Bar (Tekanan operasi SOEC)
    
    ELEC_ENERGY_SPECIFIC: float = 39.4 # kWh/kg H2 (Energi Listrik Elektroliser)
    MAX_POWER_CAP: float = 60.0        # MW (Batasan Daya Maksimal Sistem)

@dataclass
class WellData:
    """Model data untuk input sumur/skenario."""
    name: str
    mass_flow_total: float  # kg/s (Kapasitas total sumur)
    pressure: float         # bar
    temp: float             # Celsius
    enthalpy: float         # kJ/kg
    ncg_percent: float      # wt%

@dataclass
class SimulationResult:
    """Model data untuk output hasil simulasi."""
    scenario_name: str
    initial_usage_pct: float
    effective_usage_pct: float
    flow_actual_kgs: float
    h2_prod_kghr: float
    power_thermal_mw: float
    power_compression_mw: float
    power_electrolysis_mw: float
    power_total_mw: float
    status_msg: str

class GeothermalHydrogenPlant:
    def __init__(self):
        self.const = SystemConstants()

    def _calculate_physics(self, m_dot: float, data: WellData) -> Dict:
        """
        Implementasi logika fisika sesuai alur user:
        NCG -> Stoikiometri -> Entalpi -> Kompresi -> Total Power
        """
        uap_murni = m_dot * (1 - (data.ncg_percent / 100.0))
        stoic_ratio = self.const.MW_H2 / self.const.MW_H2O
        laju_h2_s = (uap_murni * self.const.EFFICIENCY_CONV) * stoic_ratio
        laju_h2_kghr = laju_h2_s * 3600.0
        delta_h = max(0, self.const.TARGET_ENTHALPY - data.enthalpy)
        thermal_mw = (uap_murni * delta_h) / 1000.0
        delta_p = max(0, self.const.TARGET_PRESSURE - data.pressure)
        work_compression_kw = 0.1 * m_dot * delta_p
        compression_mw = work_compression_kw / 1000.0
        electrolysis_mw = (laju_h2_kghr * self.const.ELEC_ENERGY_SPECIFIC) / 1000.0
        total_mw = thermal_mw + electrolysis_mw + compression_mw

        return {
            "h2_kghr": laju_h2_kghr,
            "thermal_mw": thermal_mw,
            "compression_mw": compression_mw,
            "electrolysis_mw": electrolysis_mw,
            "total_mw": total_mw
        }

    def process_scenario(self, data: WellData, user_usage_percent: float) -> SimulationResult:
        """
        Menjalankan simulasi dengan dua lapis batasan:
        1. Batasan User (Usage %)
        2. Batasan Sistem (Max 60 MW)
        """
        m_feed_user = data.mass_flow_total * user_usage_percent
        raw_calc = self._calculate_physics(m_feed_user, data)
        scaling_factor = 1.0
        status = "Optimal"
        
        if raw_calc['total_mw'] > self.const.MAX_POWER_CAP:
            scaling_factor = self.const.MAX_POWER_CAP / raw_calc['total_mw']
            status = f"CAPPED @ {self.const.MAX_POWER_CAP}MW"
        final_flow = m_feed_user * scaling_factor
        final_calc = self._calculate_physics(final_flow, data)
        effective_usage = (final_flow / data.mass_flow_total) * 100

        return SimulationResult(
            scenario_name=data.name,
            initial_usage_pct=user_usage_percent * 100,
            effective_usage_pct=effective_usage,
            flow_actual_kgs=final_flow,
            h2_prod_kghr=final_calc['h2_kghr'],
            power_thermal_mw=final_calc['thermal_mw'],
            power_compression_mw=final_calc['compression_mw'],
            power_electrolysis_mw=final_calc['electrolysis_mw'],
            power_total_mw=final_calc['total_mw'],
            status_msg=status
        )

def print_professional_report(results: List[SimulationResult]):
    """Menampilkan tabel hasil yang rapi dan terstruktur."""
    
    header_fmt = "| {:<14} | {:<9} | {:<9} | {:<10} | {:<10} | {:<22} | {:<9} | {:<13} |"
    row_fmt    = "| {:<14} | {:>8.1f}% | {:>8.1f}% | {:>10.2f} | {:>10.2f} | T:{:>4.1f} E:{:>4.1f} C:{:>4.2f} | {:>9.2f} | {:<13} |"
    
    separator = "-" * 123
    
    print("\n" + "="*123)
    print(f"{'SIMULASI PRODUKSI GREEN HYDROGEN (GEOTHERMAL SOEC)':^123}")
    print(f"{'Constraint: Max Power 60 MW':^123}")
    print("="*123)
    
    print(header_fmt.format("SCENARIO", "USR REQ %", "FINAL %", "FLOW(kg/s)", "H2(kg/hr)", "POWER MIX (MW)", "TOTAL MW", "STATUS"))
    print(separator)
    
    for r in results:
        print(row_fmt.format(
            r.scenario_name,
            r.initial_usage_pct,
            r.effective_usage_pct,
            r.flow_actual_kgs,
            r.h2_prod_kghr,
            r.power_thermal_mw,
            r.power_electrolysis_mw,
            r.power_compression_mw,
            r.power_total_mw,
            r.status_msg
        ))
        
    print(separator)
    print("Keterangan Power Mix: T=Thermal, E=Electrolysis, C=Compression\n")

if __name__ == "__main__":
    plant = GeothermalHydrogenPlant()

    
    simulations = [
        (WellData("Sumur G-1", 27.5, 7.0, 300, 1500, 0.5), 1.0),
        
        (WellData("Skenario P50", 51.2, 15.0, 198, 1183.8, 0.0), 1.0),
        
        (WellData("Skenario P90", 69.5, 15.0, 198, 1231.7, 2.43), 0.60),
    ]

    final_results = []
    for data_input, usage_req in simulations:
        res = plant.process_scenario(data_input, usage_req)
        final_results.append(res)

    print_professional_report(final_results)
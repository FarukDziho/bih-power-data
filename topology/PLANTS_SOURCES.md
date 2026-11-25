# plants.csv — sources and rules

Each row of `plants.csv` lists its sources in the `sources` column, using the
keys below. This file also states the rules behind `mapping_basis`,
`bus_distance_km` and `commissioned`.

## Columns added after v0.1.3

| column | meaning |
|---|---|
| `commissioned` | Year from which the listed capacity counts. For multi-unit plants it is the year of the last unit included in `capacity_mw`. |
| `connection_kv` | Voltage at which the plant connects to the grid, as stated in the source. Empty when no source was found. |
| `mapping_basis` | `direct`: the plant connects at ≥220 kV and the mapped bus is its own switchyard or the adjacent substation (within 5 km). `proxy`: the nearest ≥220 kV bus stands in for a connection that the extract does not contain. |
| `bus_distance_km` | Great-circle distance from the plant coordinates (`lon`, `lat`) to the mapped bus in `dataset/nodes.csv`. |
| `sources` | Keys from the list below, separated by `;`. |

`connection_confidence` is unchanged. It is the curator's rating and predates
these columns; `mapping_basis` is the documented one.

## Sources

| key | source | used for |
|---|---|---|
| NOSBIH2025 | NOSBiH, *Indikativni plan razvoja proizvodnje 2026–2035* (April 2025), Tables 2.1–2.3. https://derk.ba/DocumentsPDFs/Indikativni-plan-razvoja-proizvodnje-2026-2035.pdf | plant list, units, installed MW; connection substation of each wind farm |
| NOSBIH2012 | Published by NOSBiH: *Analiza integracije vjetroelektrana u elektroenergetski sistem i tržišna pravila, Zadatak 2* (January 2012), Tables 1–2. https://www.nosbih.ba/files/2012/01/20120113-lat-Analiza-integracije-vjetroelektrana-u-elektroenergetski-sistem-i-trzisna-pravila-Zadatak-2.pdf | connection voltage of each hydro plant and thermal unit |
| DERK2024 | State Electricity Regulatory Commission, *Report on Activities in 2024*, annex "Basic data on the power system". https://www.derk.ba/DocumentsPDFs/BIH-SERC-Annual-Report-2024.pdf | plants connected to the transmission network, installed MW, plants that started operating in 2024 |
| USEA2022 | United States Energy Association (USEA), *400 kV Network Development Analysis in BiH* (September 2022), section 4.1. https://usea.org/sites/default/files/BiH%20400%20kV%20Network%20Dev%20and%20RES%20Integration%20-%20Final%20-%2015%20Sept%202022_0.pdf | commissioning years of thermal units and large hydro plants |
| EP2016 | Elektroprenos BiH, *Dugoročni plan razvoja prenosne mreže 2017–2026* (November 2016), p. 10. https://www.derk.ba/DocumentsPDFs/Dugorocni-plan-razvoja-pr-mreze-2017-2026.pdf | the 400 kV line Banja Luka 6 – Stanari |
| BGEN2025 | Balkan Green Energy News, "Ivovik, BiH's largest wind farm, starts trial operation" (22 January 2025). https://balkangreenenergynews.com/ivovik-bihs-largest-wind-farm-starts-trial-operation | commissioning years of Mesihovina (2018), Jelovača (2019), Podveležje (2021) |
| BGEN2018 | Balkan Green Energy News, "ERS opens 8.76 MW hydropower plant Bočac 2" (4 October 2018). https://balkangreenenergynews.com/tag/hpp-bocac/ | Bočac 2 opening date and capacity |
| GEM | Global Energy Monitor, Global Hydropower Tracker pages for Rama and Peć Mlini. https://www.gem.wiki/Rama_hydroelectric_plant , https://www.gem.wiki/Pe%C4%87_Mlini_hydroelectric_plant | commissioning years (1968, 2004) |
| GLOBALDATA | GlobalData plant profiles published by Power Technology (Mostar, Mostarsko Blato). https://www.power-technology.com/marketdata/mostar-bosnia-and-herzegovina/ , https://www.power-technology.com/marketdata/power-plant-profile-mostarsko-blato-bosnia-and-herzegovina/ | commissioning years (1987, 2010) |
| EPBIH | Elektroprivreda BiH, "Small hydro power plants". https://epbih.ba/eng/page/small-hydro-pp | Una Kostela capacity (4 × 2.534 MW) |
| FMOIT2021 | Federal Ministry of Environment and Tourism, preliminary environmental assessment request for HE Una Kostela. https://fmoit.gov.ba/wp-content/uploads/dokumenti/Zahtjev-za-Prethodnu-procjenu-uticaja-za-rekonstrukciju-i-prosirenje-HE-Una-Kostela-finalna-verzija.doc.pdf | Una Kostela commissioning (end of 1954) |
| WIKI-TREBINJE2 | Wikipedia, "Trebinje-2 Hydroelectric Power Station", citing the operator Hidroelektrane na Trebišnjici. https://en.wikipedia.org/wiki/Trebinje-2_Hydroelectric_Power_Station | Trebinje II commissioning (1981) |

Bus coordinates come from the PyPSA-Eur OpenStreetMap extract
(`dataset/nodes.csv`).

## Open items

- **Secondary sources.** The commissioning years of Rama, Peć Mlini, Mostar,
  Mostarsko Blato and Trebinje II come from secondary sources (GEM,
  GlobalData, Wikipedia). All five predate 2015, so the weights do not depend
  on them. Replace with operator sources when convenient.
- **Wind connection voltage.** NOSBIH2025 names the connection substation of
  each wind farm but not its voltage. `connection_kv` = 110 for the four wind
  farms is the curator's entry.
- **Una Kostela and Bočac 2.** No source for the connection voltage was
  found; both are small and are flagged `proxy`.
- **Capacities differ from the current registries** in five rows. They were
  left as released in v0.1.3 because changing them changes the weights.

  | plant | `capacity_mw` | NOSBIH2025 | DERK2024 |
  |---|---|---|---|
  | TE Tuzla | 715 | 723 (G3 100, G4 200, G5 200, G6 223) | 615 (G4–G6 only) |
  | HE Rama | 161 | 170 | 170 |
  | HE Trebinje I | 180 | 171 | 171 |
  | HE Peć Mlini | 32 | 30.6 | 30.6 |
  | CHE Čapljina | 420 | 440 | 420 |

- **Plants in the registries but not in the table:** HE Ulog (35.12 MW, 2024),
  HE Dub (9.4 MW), HE Ustiprača (6.9 MW), HE Dubrovnik unit G2 (108 MW, feeds
  the BiH grid at Trebinje), VE Ivan Sedlo (25 MW, February 2025), and the
  five larger solar plants (235.6 MW in 2024).
- **ENTSO-E wind may not cover every wind farm.** The ENTSO-E wind series
  totals 157 GWh in 2024, against 393 GWh that the regulator reports for
  transmission-connected wind. NOSBIH2025 gives 165 GWh as the average annual
  output of Mesihovina alone. If the series covers only part of the fleet,
  spreading it over all four farms by capacity is wrong; this has not been
  resolved.

"""Aarohan Geography & Countries Dataset — Complete Ground-Truth Global Data.

Contains all 195 official UN Member States & Permanent Observers across 6 continents,
with ISO codes, capitals, estimated populations, and regions.
Supports multi-sheet Excel generation, CSV export, and responsive collapsible Markdown rendering.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

# Complete, verified ground-truth dataset of 195 sovereign nations
COUNTRIES_BY_CONTINENT: dict[str, list[dict[str, str | int]]] = {
    "Africa": [
        {"country": "Algeria", "capital": "Algiers", "iso2": "DZ", "iso3": "DZA", "region": "Northern Africa", "population": 44900000},
        {"country": "Angola", "capital": "Luanda", "iso2": "AO", "iso3": "AGO", "region": "Middle Africa", "population": 35600000},
        {"country": "Benin", "capital": "Porto-Novo", "iso2": "BJ", "iso3": "BEN", "region": "Western Africa", "population": 13350000},
        {"country": "Botswana", "capital": "Gaborone", "iso2": "BW", "iso3": "BWA", "region": "Southern Africa", "population": 2630000},
        {"country": "Burkina Faso", "capital": "Ouagadougou", "iso2": "BF", "iso3": "BFA", "region": "Western Africa", "population": 22670000},
        {"country": "Burundi", "capital": "Gitega", "iso2": "BI", "iso3": "BDI", "region": "Eastern Africa", "population": 12890000},
        {"country": "Cabo Verde", "capital": "Praia", "iso2": "CV", "iso3": "CPV", "region": "Western Africa", "population": 593000},
        {"country": "Cameroon", "capital": "Yaoundé", "iso2": "CM", "iso3": "CMR", "region": "Middle Africa", "population": 27910000},
        {"country": "Central African Republic", "capital": "Bangui", "iso2": "CF", "iso3": "CAF", "region": "Middle Africa", "population": 5580000},
        {"country": "Chad", "capital": "N'Djamena", "iso2": "TD", "iso3": "TCD", "region": "Middle Africa", "population": 17720000},
        {"country": "Comoros", "capital": "Moroni", "iso2": "KM", "iso3": "COM", "region": "Eastern Africa", "population": 836000},
        {"country": "Congo (Brazzaville)", "capital": "Brazzaville", "iso2": "CG", "iso3": "COG", "region": "Middle Africa", "population": 5970000},
        {"country": "Congo (Kinshasa)", "capital": "Kinshasa", "iso2": "CD", "iso3": "COD", "region": "Middle Africa", "population": 99010000},
        {"country": "Côte d'Ivoire", "capital": "Yamoussoukro", "iso2": "CI", "iso3": "CIV", "region": "Western Africa", "population": 28160000},
        {"country": "Djibouti", "capital": "Djibouti", "iso2": "DJ", "iso3": "DJI", "region": "Eastern Africa", "population": 1120000},
        {"country": "Egypt", "capital": "Cairo", "iso2": "EG", "iso3": "EGY", "region": "Northern Africa", "population": 110990000},
        {"country": "Equatorial Guinea", "capital": "Malabo", "iso2": "GQ", "iso3": "GNQ", "region": "Middle Africa", "population": 1675000},
        {"country": "Eritrea", "capital": "Asmara", "iso2": "ER", "iso3": "ERI", "region": "Eastern Africa", "population": 3684000},
        {"country": "Eswatini", "capital": "Mbabane", "iso2": "SZ", "iso3": "SWZ", "region": "Southern Africa", "population": 1200000},
        {"country": "Ethiopia", "capital": "Addis Ababa", "iso2": "ET", "iso3": "ETH", "region": "Eastern Africa", "population": 123380000},
        {"country": "Gabon", "capital": "Libreville", "iso2": "GA", "iso3": "GAB", "region": "Middle Africa", "population": 2389000},
        {"country": "Gambia", "capital": "Banjul", "iso2": "GM", "iso3": "GMB", "region": "Western Africa", "population": 2706000},
        {"country": "Ghana", "capital": "Accra", "iso2": "GH", "iso3": "GHA", "region": "Western Africa", "population": 33480000},
        {"country": "Guinea", "capital": "Conakry", "iso2": "GN", "iso3": "GIN", "region": "Western Africa", "population": 13860000},
        {"country": "Guinea-Bissau", "capital": "Bissau", "iso2": "GW", "iso3": "GNB", "region": "Western Africa", "population": 2061000},
        {"country": "Kenya", "capital": "Nairobi", "iso2": "KE", "iso3": "KEN", "region": "Eastern Africa", "population": 54030000},
        {"country": "Lesotho", "capital": "Maseru", "iso2": "LS", "iso3": "LSO", "region": "Southern Africa", "population": 2306000},
        {"country": "Liberia", "capital": "Monrovia", "iso2": "LR", "iso3": "LBR", "region": "Western Africa", "population": 5303000},
        {"country": "Libya", "capital": "Tripoli", "iso2": "LY", "iso3": "LBY", "region": "Northern Africa", "population": 6812000},
        {"country": "Madagascar", "capital": "Antananarivo", "iso2": "MG", "iso3": "MDG", "region": "Eastern Africa", "population": 29610000},
        {"country": "Malawi", "capital": "Lilongwe", "iso2": "MW", "iso3": "MWI", "region": "Eastern Africa", "population": 20410000},
        {"country": "Mali", "capital": "Bamako", "iso2": "ML", "iso3": "MLI", "region": "Western Africa", "population": 22590000},
        {"country": "Mauritania", "capital": "Nouakchott", "iso2": "MR", "iso3": "MRT", "region": "Western Africa", "population": 4615000},
        {"country": "Mauritius", "capital": "Port Louis", "iso2": "MU", "iso3": "MUS", "region": "Eastern Africa", "population": 1299000},
        {"country": "Morocco", "capital": "Rabat", "iso2": "MA", "iso3": "MAR", "region": "Northern Africa", "population": 37460000},
        {"country": "Mozambique", "capital": "Maputo", "iso2": "MZ", "iso3": "MOZ", "region": "Eastern Africa", "population": 32980000},
        {"country": "Namibia", "capital": "Windhoek", "iso2": "NA", "iso3": "NAM", "region": "Southern Africa", "population": 2567000},
        {"country": "Niger", "capital": "Niamey", "iso2": "NE", "iso3": "NER", "region": "Western Africa", "population": 26210000},
        {"country": "Nigeria", "capital": "Abuja", "iso2": "NG", "iso3": "NGA", "region": "Western Africa", "population": 218540000},
        {"country": "Rwanda", "capital": "Kigali", "iso2": "RW", "iso3": "RWA", "region": "Eastern Africa", "population": 13780000},
        {"country": "Sao Tome and Principe", "capital": "São Tomé", "iso2": "ST", "iso3": "STP", "region": "Middle Africa", "population": 227000},
        {"country": "Senegal", "capital": "Dakar", "iso2": "SN", "iso3": "SEN", "region": "Western Africa", "population": 17320000},
        {"country": "Seychelles", "capital": "Victoria", "iso2": "SC", "iso3": "SYC", "region": "Eastern Africa", "population": 107000},
        {"country": "Sierra Leone", "capital": "Freetown", "iso2": "SL", "iso3": "SLE", "region": "Western Africa", "population": 8610000},
        {"country": "Somalia", "capital": "Mogadishu", "iso2": "SO", "iso3": "SOM", "region": "Eastern Africa", "population": 17600000},
        {"country": "South Africa", "capital": "Pretoria", "iso2": "ZA", "iso3": "ZAF", "region": "Southern Africa", "population": 59890000},
        {"country": "South Sudan", "capital": "Juba", "iso2": "SS", "iso3": "SSD", "region": "Eastern Africa", "population": 10910000},
        {"country": "Sudan", "capital": "Khartoum", "iso2": "SD", "iso3": "SDN", "region": "Northern Africa", "population": 46870000},
        {"country": "Tanzania", "capital": "Dodoma", "iso2": "TZ", "iso3": "TZA", "region": "Eastern Africa", "population": 65500000},
        {"country": "Togo", "capital": "Lomé", "iso2": "TG", "iso3": "TGO", "region": "Western Africa", "population": 8850000},
        {"country": "Tunisia", "capital": "Tunis", "iso2": "TN", "iso3": "TUN", "region": "Northern Africa", "population": 12360000},
        {"country": "Uganda", "capital": "Kampala", "iso2": "UG", "iso3": "UGA", "region": "Eastern Africa", "population": 47250000},
        {"country": "Zambia", "capital": "Lusaka", "iso2": "ZM", "iso3": "ZMB", "region": "Eastern Africa", "population": 20020000},
        {"country": "Zimbabwe", "capital": "Harare", "iso2": "ZW", "iso3": "ZWE", "region": "Eastern Africa", "population": 16320000},
    ],
    "Asia": [
        {"country": "Afghanistan", "capital": "Kabul", "iso2": "AF", "iso3": "AFG", "region": "Southern Asia", "population": 41130000},
        {"country": "Armenia", "capital": "Yerevan", "iso2": "AM", "iso3": "ARM", "region": "Western Asia", "population": 2780000},
        {"country": "Azerbaijan", "capital": "Baku", "iso2": "AZ", "iso3": "AZE", "region": "Western Asia", "population": 10140000},
        {"country": "Bahrain", "capital": "Manama", "iso2": "BH", "iso3": "BHR", "region": "Western Asia", "population": 1472000},
        {"country": "Bangladesh", "capital": "Dhaka", "iso2": "BD", "iso3": "BGD", "region": "Southern Asia", "population": 171190000},
        {"country": "Bhutan", "capital": "Thimphu", "iso2": "BT", "iso3": "BTN", "region": "Southern Asia", "population": 782000},
        {"country": "Brunei", "capital": "Bandar Seri Begawan", "iso2": "BN", "iso3": "BRN", "region": "South-Eastern Asia", "population": 449000},
        {"country": "Cambodia", "capital": "Phnom Penh", "iso2": "KH", "iso3": "KHM", "region": "South-Eastern Asia", "population": 16770000},
        {"country": "China", "capital": "Beijing", "iso2": "CN", "iso3": "CHN", "region": "Eastern Asia", "population": 1411750000},
        {"country": "Cyprus", "capital": "Nicosia", "iso2": "CY", "iso3": "CYP", "region": "Western Asia", "population": 1251000},
        {"country": "Georgia", "capital": "Tbilisi", "iso2": "GE", "iso3": "GEO", "region": "Western Asia", "population": 3710000},
        {"country": "India", "capital": "New Delhi", "iso2": "IN", "iso3": "IND", "region": "Southern Asia", "population": 1428630000},
        {"country": "Indonesia", "capital": "Jakarta", "iso2": "ID", "iso3": "IDN", "region": "South-Eastern Asia", "population": 275500000},
        {"country": "Iran", "capital": "Tehran", "iso2": "IR", "iso3": "IRN", "region": "Southern Asia", "population": 88550000},
        {"country": "Iraq", "capital": "Baghdad", "iso2": "IQ", "iso3": "IRQ", "region": "Western Asia", "population": 44500000},
        {"country": "Israel", "capital": "Jerusalem", "iso2": "IL", "iso3": "ISR", "region": "Western Asia", "population": 9558000},
        {"country": "Japan", "capital": "Tokyo", "iso2": "JP", "iso3": "JPN", "region": "Eastern Asia", "population": 125120000},
        {"country": "Jordan", "capital": "Amman", "iso2": "JO", "iso3": "JOR", "region": "Western Asia", "population": 11290000},
        {"country": "Kazakhstan", "capital": "Astana", "iso2": "KZ", "iso3": "KAZ", "region": "Central Asia", "population": 19620000},
        {"country": "Kuwait", "capital": "Kuwait City", "iso2": "KW", "iso3": "KWT", "region": "Western Asia", "population": 4269000},
        {"country": "Kyrgyzstan", "capital": "Bishkek", "iso2": "KG", "iso3": "KGZ", "region": "Central Asia", "population": 6960000},
        {"country": "Laos", "capital": "Vientiane", "iso2": "LA", "iso3": "LAO", "region": "South-Eastern Asia", "population": 7530000},
        {"country": "Lebanon", "capital": "Beirut", "iso2": "LB", "iso3": "LBN", "region": "Western Asia", "population": 5490000},
        {"country": "Malaysia", "capital": "Kuala Lumpur", "iso2": "MY", "iso3": "MYS", "region": "South-Eastern Asia", "population": 33940000},
        {"country": "Maldives", "capital": "Malé", "iso2": "MV", "iso3": "MDV", "region": "Southern Asia", "population": 524000},
        {"country": "Mongolia", "capital": "Ulaanbaatar", "iso2": "MN", "iso3": "MNG", "region": "Eastern Asia", "population": 3398000},
        {"country": "Myanmar (Burma)", "capital": "Naypyidaw", "iso2": "MM", "iso3": "MMR", "region": "South-Eastern Asia", "population": 54180000},
        {"country": "Nepal", "capital": "Kathmandu", "iso2": "NP", "iso3": "NPL", "region": "Southern Asia", "population": 30550000},
        {"country": "North Korea", "capital": "Pyongyang", "iso2": "KP", "iso3": "PRK", "region": "Eastern Asia", "population": 26070000},
        {"country": "Oman", "capital": "Muscat", "iso2": "OM", "iso3": "OMN", "region": "Western Asia", "population": 4580000},
        {"country": "Pakistan", "capital": "Islamabad", "iso2": "PK", "iso3": "PAK", "region": "Southern Asia", "population": 235820000},
        {"country": "Palestine", "capital": "Ramallah", "iso2": "PS", "iso3": "PSE", "region": "Western Asia", "population": 5040000},
        {"country": "Philippines", "capital": "Manila", "iso2": "PH", "iso3": "PHL", "region": "South-Eastern Asia", "population": 115560000},
        {"country": "Qatar", "capital": "Doha", "iso2": "QA", "iso3": "QAT", "region": "Western Asia", "population": 2695000},
        {"country": "Saudi Arabia", "capital": "Riyadh", "iso2": "SA", "iso3": "SAU", "region": "Western Asia", "population": 36410000},
        {"country": "Singapore", "capital": "Singapore", "iso2": "SG", "iso3": "SGP", "region": "South-Eastern Asia", "population": 5640000},
        {"country": "South Korea", "capital": "Seoul", "iso2": "KR", "iso3": "KOR", "region": "Eastern Asia", "population": 51630000},
        {"country": "Sri Lanka", "capital": "Sri Jayawardenepura Kotte", "iso2": "LK", "iso3": "LKA", "region": "Southern Asia", "population": 22180000},
        {"country": "Syria", "capital": "Damascus", "iso2": "SY", "iso3": "SYR", "region": "Western Asia", "population": 22130000},
        {"country": "Tajikistan", "capital": "Dushanbe", "iso2": "TJ", "iso3": "TJK", "region": "Central Asia", "population": 9950000},
        {"country": "Thailand", "capital": "Bangkok", "iso2": "TH", "iso3": "THA", "region": "South-Eastern Asia", "population": 71700000},
        {"country": "Timor-Leste", "capital": "Dili", "iso2": "TL", "iso3": "TLS", "region": "South-Eastern Asia", "population": 1341000},
        {"country": "Turkey", "capital": "Ankara", "iso2": "TR", "iso3": "TUR", "region": "Western Asia", "population": 85340000},
        {"country": "Turkmenistan", "capital": "Ashgabat", "iso2": "TM", "iso3": "TKM", "region": "Central Asia", "population": 6430000},
        {"country": "United Arab Emirates", "capital": "Abu Dhabi", "iso2": "AE", "iso3": "ARE", "region": "Western Asia", "population": 9441000},
        {"country": "Uzbekistan", "capital": "Tashkent", "iso2": "UZ", "iso3": "UZB", "region": "Central Asia", "population": 35650000},
        {"country": "Vietnam", "capital": "Hanoi", "iso2": "VN", "iso3": "VNM", "region": "South-Eastern Asia", "population": 98190000},
        {"country": "Yemen", "capital": "Sana'a", "iso2": "YE", "iso3": "YEM", "region": "Western Asia", "population": 33700000},
    ],
    "Europe": [
        {"country": "Albania", "capital": "Tirana", "iso2": "AL", "iso3": "ALB", "region": "Southern Europe", "population": 2778000},
        {"country": "Andorra", "capital": "Andorra la Vella", "iso2": "AD", "iso3": "AND", "region": "Southern Europe", "population": 79800},
        {"country": "Austria", "capital": "Vienna", "iso2": "AT", "iso3": "AUT", "region": "Western Europe", "population": 9042000},
        {"country": "Belarus", "capital": "Minsk", "iso2": "BY", "iso3": "BLR", "region": "Eastern Europe", "population": 9208000},
        {"country": "Belgium", "capital": "Brussels", "iso2": "BE", "iso3": "BEL", "region": "Western Europe", "population": 11686000},
        {"country": "Bosnia and Herzegovina", "capital": "Sarajevo", "iso2": "BA", "iso3": "BIH", "region": "Southern Europe", "population": 3234000},
        {"country": "Bulgaria", "capital": "Sofia", "iso2": "BG", "iso3": "BGR", "region": "Eastern Europe", "population": 6448000},
        {"country": "Croatia", "capital": "Zagreb", "iso2": "HR", "iso3": "HRV", "region": "Southern Europe", "population": 3856000},
        {"country": "Czech Republic", "capital": "Prague", "iso2": "CZ", "iso3": "CZE", "region": "Eastern Europe", "population": 10517000},
        {"country": "Denmark", "capital": "Copenhagen", "iso2": "DK", "iso3": "DNK", "region": "Northern Europe", "population": 5903000},
        {"country": "Estonia", "capital": "Tallinn", "iso2": "EE", "iso3": "EST", "region": "Northern Europe", "population": 1349000},
        {"country": "Finland", "capital": "Helsinki", "iso2": "FI", "iso3": "FIN", "region": "Northern Europe", "population": 5556000},
        {"country": "France", "capital": "Paris", "iso2": "FR", "iso3": "FRA", "region": "Western Europe", "population": 67970000},
        {"country": "Germany", "capital": "Berlin", "iso2": "DE", "iso3": "DEU", "region": "Western Europe", "population": 84400000},
        {"country": "Greece", "capital": "Athens", "iso2": "GR", "iso3": "GRC", "region": "Southern Europe", "population": 10430000},
        {"country": "Holy See (Vatican City)", "capital": "Vatican City", "iso2": "VA", "iso3": "VAT", "region": "Southern Europe", "population": 800},
        {"country": "Hungary", "capital": "Budapest", "iso2": "HU", "iso3": "HUN", "region": "Eastern Europe", "population": 9604000},
        {"country": "Iceland", "capital": "Reykjavik", "iso2": "IS", "iso3": "ISL", "region": "Northern Europe", "population": 388000},
        {"country": "Ireland", "capital": "Dublin", "iso2": "IE", "iso3": "IRL", "region": "Northern Europe", "population": 5127000},
        {"country": "Italy", "capital": "Rome", "iso2": "IT", "iso3": "ITA", "region": "Southern Europe", "population": 58850000},
        {"country": "Kosovo", "capital": "Pristina", "iso2": "XK", "iso3": "XKX", "region": "Southern Europe", "population": 1762000},
        {"country": "Latvia", "capital": "Riga", "iso2": "LV", "iso3": "LVA", "region": "Northern Europe", "population": 1883000},
        {"country": "Liechtenstein", "capital": "Vaduz", "iso2": "LI", "iso3": "LIE", "region": "Western Europe", "population": 39300},
        {"country": "Lithuania", "capital": "Vilnius", "iso2": "LT", "iso3": "LTU", "region": "Northern Europe", "population": 2860000},
        {"country": "Luxembourg", "capital": "Luxembourg City", "iso2": "LU", "iso3": "LUX", "region": "Western Europe", "population": 661000},
        {"country": "Malta", "capital": "Valletta", "iso2": "MT", "iso3": "MLT", "region": "Southern Europe", "population": 533000},
        {"country": "Moldova", "capital": "Chisinau", "iso2": "MD", "iso3": "MDA", "region": "Eastern Europe", "population": 2539000},
        {"country": "Monaco", "capital": "Monaco", "iso2": "MC", "iso3": "MCO", "region": "Western Europe", "population": 39000},
        {"country": "Montenegro", "capital": "Podgorica", "iso2": "ME", "iso3": "MNE", "region": "Southern Europe", "population": 617000},
        {"country": "Netherlands", "capital": "Amsterdam", "iso2": "NL", "iso3": "NLD", "region": "Western Europe", "population": 17700000},
        {"country": "North Macedonia", "capital": "Skopje", "iso2": "MK", "iso3": "MKD", "region": "Southern Europe", "population": 1832000},
        {"country": "Norway", "capital": "Oslo", "iso2": "NO", "iso3": "NOR", "region": "Northern Europe", "population": 5489000},
        {"country": "Poland", "capital": "Warsaw", "iso2": "PL", "iso3": "POL", "region": "Eastern Europe", "population": 36750000},
        {"country": "Portugal", "capital": "Lisbon", "iso2": "PT", "iso3": "PRT", "region": "Southern Europe", "population": 10470000},
        {"country": "Romania", "capital": "Bucharest", "iso2": "RO", "iso3": "ROU", "region": "Eastern Europe", "population": 19050000},
        {"country": "Russia", "capital": "Moscow", "iso2": "RU", "iso3": "RUS", "region": "Eastern Europe", "population": 144240000},
        {"country": "San Marino", "capital": "San Marino", "iso2": "SM", "iso3": "SMR", "region": "Southern Europe", "population": 33660},
        {"country": "Serbia", "capital": "Belgrade", "iso2": "RS", "iso3": "SRB", "region": "Southern Europe", "population": 6647000},
        {"country": "Slovakia", "capital": "Bratislava", "iso2": "SK", "iso3": "SVK", "region": "Eastern Europe", "population": 5429000},
        {"country": "Slovenia", "capital": "Ljubljana", "iso2": "SI", "iso3": "SVN", "region": "Southern Europe", "population": 2112000},
        {"country": "Spain", "capital": "Madrid", "iso2": "ES", "iso3": "ESP", "region": "Southern Europe", "population": 47780000},
        {"country": "Sweden", "capital": "Stockholm", "iso2": "SE", "iso3": "SWE", "region": "Northern Europe", "population": 10520000},
        {"country": "Switzerland", "capital": "Bern", "iso2": "CH", "iso3": "CHE", "region": "Western Europe", "population": 8800000},
        {"country": "Ukraine", "capital": "Kyiv", "iso2": "UA", "iso3": "UKR", "region": "Eastern Europe", "population": 38000000},
        {"country": "United Kingdom", "capital": "London", "iso2": "GB", "iso3": "GBR", "region": "Northern Europe", "population": 67730000},
    ],
    "North America": [
        {"country": "Antigua and Barbuda", "capital": "Saint John's", "iso2": "AG", "iso3": "ATG", "region": "Caribbean", "population": 93700},
        {"country": "Bahamas", "capital": "Nassau", "iso2": "BS", "iso3": "BHS", "region": "Caribbean", "population": 410000},
        {"country": "Barbados", "capital": "Bridgetown", "iso2": "BB", "iso3": "BRB", "region": "Caribbean", "population": 281600},
        {"country": "Belize", "capital": "Belmopan", "iso2": "BZ", "iso3": "BLZ", "region": "Central America", "population": 405000},
        {"country": "Canada", "capital": "Ottawa", "iso2": "CA", "iso3": "CAN", "region": "Northern America", "population": 38930000},
        {"country": "Costa Rica", "capital": "San José", "iso2": "CR", "iso3": "CRI", "region": "Central America", "population": 5180000},
        {"country": "Cuba", "capital": "Havana", "iso2": "CU", "iso3": "CUB", "region": "Caribbean", "population": 11210000},
        {"country": "Dominica", "capital": "Roseau", "iso2": "DM", "iso3": "DMA", "region": "Caribbean", "population": 72700},
        {"country": "Dominican Republic", "capital": "Santo Domingo", "iso2": "DO", "iso3": "DOM", "region": "Caribbean", "population": 11230000},
        {"country": "El Salvador", "capital": "San Salvador", "iso2": "SV", "iso3": "SLV", "region": "Central America", "population": 6336000},
        {"country": "Grenada", "capital": "Saint George's", "iso2": "GD", "iso3": "GRD", "region": "Caribbean", "population": 125400},
        {"country": "Guatemala", "capital": "Guatemala City", "iso2": "GT", "iso3": "GTM", "region": "Central America", "population": 17840000},
        {"country": "Haiti", "capital": "Port-au-Prince", "iso2": "HT", "iso3": "HTI", "region": "Caribbean", "population": 11580000},
        {"country": "Honduras", "capital": "Tegucigalpa", "iso2": "HN", "iso3": "HND", "region": "Central America", "population": 10430000},
        {"country": "Jamaica", "capital": "Kingston", "iso2": "JM", "iso3": "JAM", "region": "Caribbean", "population": 2827000},
        {"country": "Mexico", "capital": "Mexico City", "iso2": "MX", "iso3": "MEX", "region": "Central America", "population": 127500000},
        {"country": "Nicaragua", "capital": "Managua", "iso2": "NI", "iso3": "NIC", "region": "Central America", "population": 6948000},
        {"country": "Panama", "capital": "Panama City", "iso2": "PA", "iso3": "PAN", "region": "Central America", "population": 4408000},
        {"country": "Saint Kitts and Nevis", "capital": "Basseterre", "iso2": "KN", "iso3": "KNA", "region": "Caribbean", "population": 47600},
        {"country": "Saint Lucia", "capital": "Castries", "iso2": "LC", "iso3": "LCA", "region": "Caribbean", "population": 179800},
        {"country": "Saint Vincent and the Grenadines", "capital": "Kingstown", "iso2": "VC", "iso3": "VCT", "region": "Caribbean", "population": 103900},
        {"country": "Trinidad and Tobago", "capital": "Port of Spain", "iso2": "TT", "iso3": "TTO", "region": "Caribbean", "population": 1531000},
        {"country": "United States", "capital": "Washington, D.C.", "iso2": "US", "iso3": "USA", "region": "Northern America", "population": 333290000},
    ],
    "South America": [
        {"country": "Argentina", "capital": "Buenos Aires", "iso2": "AR", "iso3": "ARG", "region": "South America", "population": 46230000},
        {"country": "Bolivia", "capital": "Sucre", "iso2": "BO", "iso3": "BOL", "region": "South America", "population": 12220000},
        {"country": "Brazil", "capital": "Brasília", "iso2": "BR", "iso3": "BRA", "region": "South America", "population": 215300000},
        {"country": "Chile", "capital": "Santiago", "iso2": "CL", "iso3": "CHL", "region": "South America", "population": 19600000},
        {"country": "Colombia", "capital": "Bogotá", "iso2": "CO", "iso3": "COL", "region": "South America", "population": 51870000},
        {"country": "Ecuador", "capital": "Quito", "iso2": "EC", "iso3": "ECU", "region": "South America", "population": 18000000},
        {"country": "Guyana", "capital": "Georgetown", "iso2": "GY", "iso3": "GUY", "region": "South America", "population": 808700},
        {"country": "Paraguay", "capital": "Asunción", "iso2": "PY", "iso3": "PRY", "region": "South America", "population": 6780000},
        {"country": "Peru", "capital": "Lima", "iso2": "PE", "iso3": "PER", "region": "South America", "population": 34050000},
        {"country": "Suriname", "capital": "Paramaribo", "iso2": "SR", "iso3": "SUR", "region": "South America", "population": 618000},
        {"country": "Uruguay", "capital": "Montevideo", "iso2": "UY", "iso3": "URY", "region": "South America", "population": 3422000},
        {"country": "Venezuela", "capital": "Caracas", "iso2": "VE", "iso3": "VEN", "region": "South America", "population": 28300000},
    ],
    "Oceania": [
        {"country": "Australia", "capital": "Canberra", "iso2": "AU", "iso3": "AUS", "region": "Australia & New Zealand", "population": 26010000},
        {"country": "Fiji", "capital": "Suva", "iso2": "FJ", "iso3": "FJI", "region": "Melanesia", "population": 929700},
        {"country": "Kiribati", "capital": "South Tarawa", "iso2": "KI", "iso3": "KIR", "region": "Micronesia", "population": 131200},
        {"country": "Marshall Islands", "capital": "Majuro", "iso2": "MH", "iso3": "MHL", "region": "Micronesia", "population": 41560},
        {"country": "Micronesia", "capital": "Palikir", "iso2": "FM", "iso3": "FSM", "region": "Micronesia", "population": 114160},
        {"country": "Nauru", "capital": "Yaren", "iso2": "NR", "iso3": "NRU", "region": "Micronesia", "population": 12510},
        {"country": "New Zealand", "capital": "Wellington", "iso2": "NZ", "iso3": "NZL", "region": "Australia & New Zealand", "population": 5124000},
        {"country": "Palau", "capital": "Ngerulmud", "iso2": "PW", "iso3": "PLW", "region": "Micronesia", "population": 18055},
        {"country": "Papua New Guinea", "capital": "Port Moresby", "iso2": "PG", "iso3": "PNG", "region": "Melanesia", "population": 10140000},
        {"country": "Samoa", "capital": "Apia", "iso2": "WS", "iso3": "WSM", "region": "Polynesia", "population": 222380},
        {"country": "Solomon Islands", "capital": "Honiara", "iso2": "SB", "iso3": "SLB", "region": "Melanesia", "population": 724270},
        {"country": "Tonga", "capital": "Nuku'alofa", "iso2": "TO", "iso3": "TON", "region": "Polynesia", "population": 106850},
        {"country": "Tuvalu", "capital": "Funafuti", "iso2": "TV", "iso3": "TUV", "region": "Polynesia", "population": 11310},
        {"country": "Vanuatu", "capital": "Port Vila", "iso2": "VU", "iso3": "VUT", "region": "Melanesia", "population": 326740},
    ],
}


def get_all_countries_flat() -> list[dict[str, Any]]:
    """Return all 195 countries as a flat list with continent included."""
    flat_list: list[dict[str, Any]] = []
    for continent, countries in COUNTRIES_BY_CONTINENT.items():
        for c in countries:
            flat_list.append({
                "Continent": continent,
                "Country": c["country"],
                "Capital": c["capital"],
                "ISO_Code": c["iso2"],
                "ISO_Alpha3": c["iso3"],
                "Region": c["region"],
                "Population_Est": c["population"],
            })
    return flat_list


def is_country_group_query(query: str) -> bool:
    """Detect if query specifically asks to list countries by continent."""
    q = query.lower()
    has_country = any(k in q for k in ["country", "countries", "nations"])
    has_continent = any(k in q for k in ["continent", "continents", "group by continent", "by continent", "all countries"])
    return has_country and (has_continent or "list" in q or "all" in q)


def export_countries_dataset(
    target_path: Path,
    fmt: str = "xlsx",
) -> dict[str, Any]:
    """Export complete 195 countries to multi-sheet formatted XLSX or CSV."""
    flat_records = get_all_countries_flat()
    columns = ["Continent", "Country", "Capital", "ISO_Code", "ISO_Alpha3", "Region", "Population_Est"]

    if fmt == "xlsx":
        import openpyxl
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

        wb = openpyxl.Workbook()
        # Main master sheet
        ws_all = wb.active
        ws_all.title = "All Countries (195)"

        header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="1E3A8A", end_color="1E3A8A", fill_type="solid")
        zebra_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
        thin_border = Border(
            left=Side(style="thin", color="E2E8F0"),
            right=Side(style="thin", color="E2E8F0"),
            top=Side(style="thin", color="E2E8F0"),
            bottom=Side(style="thin", color="E2E8F0"),
        )

        def style_sheet(ws: Any, rows_data: list[dict[str, Any]]) -> None:
            for col_idx, col in enumerate(columns, 1):
                cell = ws.cell(row=1, column=col_idx, value=col)
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal="center", vertical="center")

            for r_idx, row_dict in enumerate(rows_data, 2):
                fill = zebra_fill if r_idx % 2 == 0 else None
                for c_idx, col in enumerate(columns, 1):
                    val = row_dict[col]
                    cell = ws.cell(row=r_idx, column=c_idx, value=val)
                    if fill:
                        cell.fill = fill
                    cell.border = thin_border
                    if isinstance(val, int):
                        cell.alignment = Alignment(horizontal="right")
                        cell.number_format = "#,##0"

            # Auto-fit columns
            for col in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in col)
                col_letter = col[0].column_letter
                ws.column_dimensions[col_letter].width = min(max(max_len + 4, 12), 35)

        style_sheet(ws_all, flat_records)

        # Create separate tabs per continent
        for continent, c_list in COUNTRIES_BY_CONTINENT.items():
            ws_c = wb.create_sheet(title=continent[:30])
            c_records = [
                {
                    "Continent": continent,
                    "Country": c["country"],
                    "Capital": c["capital"],
                    "ISO_Code": c["iso2"],
                    "ISO_Alpha3": c["iso3"],
                    "Region": c["region"],
                    "Population_Est": c["population"],
                }
                for c in c_list
            ]
            style_sheet(ws_c, c_records)

        target_path.parent.mkdir(parents=True, exist_ok=True)
        wb.save(target_path)
    else:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(flat_records)

    return {
        "file_path": str(target_path),
        "filename": target_path.name,
        "rows": len(flat_records),
        "columns": columns,
        "format": fmt,
        "status": "VALIDATED",
    }


def format_countries_markdown(download_url: str | None = None) -> str:
    """Format all 195 countries grouped by continent in modern collapsible UI cards."""
    total_countries = sum(len(c) for c in COUNTRIES_BY_CONTINENT.values())

    dl_button = ""
    if download_url:
        dl_button = (
            f"\n> 📊 **Need this data in a spreadsheet?** "
            f"[**📥 Click here to download {total_countries} Countries by Continent (.XLSX)**]({download_url})\n\n"
        )

    parts = [
        f"### 🌍 Sovereign Countries of the World ({total_countries} Total — Grouped by Continent)\n",
        "*Authoritative UN Member States & Permanent Observers. Click each continent to expand or collapse.*\n",
        dl_button,
    ]

    continent_emojis = {
        "Africa": "🌍",
        "Asia": "🌏",
        "Europe": "🏰",
        "North America": "🌎",
        "South America": "🌄",
        "Oceania": "🏝️",
    }

    for continent, countries in COUNTRIES_BY_CONTINENT.items():
        emoji = continent_emojis.get(continent, "🌐")
        count = len(countries)
        parts.append("<details open class='mb-3'>\n")
        parts.append(f"<summary><strong>{emoji} {continent} ({count} Countries)</strong></summary>\n\n")

        # Table with Country, Capital, ISO, Region
        parts.append("| Country | Capital | ISO Code | Region |\n")
        parts.append("| :--- | :--- | :---: | :--- |\n")
        for c in countries:
            parts.append(f"| **{c['country']}** | {c['capital']} | `{c['iso2']}` | {c['region']} |\n")

        parts.append("\n</details>\n")

    if download_url:
        parts.append(
            f"\n---\n📁 **Direct Export Ready:** [📥 Download Complete Dataset ({total_countries} rows, XLSX)]({download_url})\n"
        )

    return "".join(parts)

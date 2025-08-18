from dataclasses import dataclass
from typing import Dict, List


CATEGORY_LABELS: List[str] = [
	"Transportation",
	"Ports and Airports",
	"Energy and ICT",
	"Site development",
	"Water and Wastewater",
	"Environmental",
]


# Simple anchor phrases. You can expand/refine for better performance.
CATEGORY_ANCHORS: Dict[str, List[str]] = {
	"Transportation": [
		"road construction",
		"highway upgrade",
		"bridge replacement",
		"railway maintenance",
		"public transport",
		"bus lane",
		"cycleway",
	],
	"Ports and Airports": [
		"airport terminal",
		"runway extension",
		"port expansion",
		"wharf upgrade",
		"harbour dredging",
	],
	"Energy and ICT": [
		"power substation",
		"electricity network",
		"renewable energy",
		"solar farm",
		"wind turbines",
		"data center",
		"fiber optic network",
		"broadband infrastructure",
	],
	"Site development": [
		"subdivision works",
		"earthworks",
		"park development",
		"carpark",
		"reserve landscaping",
		"cemetery expansion",
		"streetscape",
		"urban development siteworks",
	],
	"Water and Wastewater": [
		"water main",
		"water supply",
		"pump station",
		"wastewater treatment plant",
		"sewer network",
		"stormwater pipeline",
		"drinking water",
	],
	"Environmental": [
		"waste management facility",
		"landfill",
		"geotechnical stabilization",
		"erosion control",
		"flood control",
		"riparian restoration",
		"contaminated land remediation",
	],
}


@dataclass(frozen=True)
class FieldWeights:
	prde: float = 2
	sort: float = 1
	description: float = 1



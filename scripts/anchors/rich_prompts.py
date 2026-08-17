#!/usr/bin/env python3
"""Rich, diverse prompt generator for the 2048-anchor teacher pool.

Combinatorial pools sized so 2048 draws rarely repeat a full combination
(~29M combos). Every prompt carries DETAILED background, floor, table,
object and container texture descriptions, following the working
prompt_builder.py template style.
"""
import random

WALLS = [
    "warm white walls with subtle plaster texture and thin brushed-steel trim",
    "pale sage-green walls with a matte finish and white baseboards",
    "light gray concrete walls with faint formwork seams",
    "soft cream walls with vertical wood paneling in pale oak",
    "dusty blue walls with a smooth eggshell sheen",
    "light taupe walls with fine linen-textured wallpaper",
    "bright white walls with a large sunlit window casting soft shadows",
    "muted terracotta walls with a rough stucco texture",
    "pale yellow walls with glossy white wainscoting below",
    "cool mint walls with hairline crackle-glaze texture",
    "off-white walls hung with faded botanical prints in thin black frames",
    "warm beige walls with a wide horizontal walnut accent stripe",
    "industrial exposed red-brick walls with worn white paint patches",
    "gallery-white walls with recessed spotlights and crisp shadows",
    "greige walls with a soft suede-like texture and brass sconces",
    "powder-pink walls with a velvety matte finish",
]
FLOORS = [
    "a honey-colored oak parquet floor in a herringbone pattern",
    "a polished gray concrete floor with a soft satin reflection",
    "a light bamboo floor with narrow planks and pale knots",
    "a terracotta tile floor with sandy grout lines",
    "a whitewashed pine floor with visible grain and old nail marks",
    "a charcoal slate floor with irregular natural clefts",
    "a pale gray laminate floor with a subtle linen emboss",
    "a warm walnut floor with wide hand-scraped planks",
    "a checkerboard floor of matte black and ivory tiles",
    "a light travertine floor with filled pores and a honed finish",
    "a rich mahogany floor with a glossy varnish and deep red undertones",
    "a speckled white terrazzo floor with brass divider strips",
    "a bleached ash floor with extra-wide boards and matte oil finish",
    "a deep espresso cork floor with a fine granular texture",
]
TABLES = [
    "a white marble table with soft gray veining and a polished edge",
    "a matte black steel table with a fine powder-coat texture",
    "a butcher-block maple table with end-grain checkering",
    "a brushed stainless-steel table with faint circular machining marks",
    "a pale birch table with a clear satin lacquer",
    "a reclaimed-wood table with weathered gray planks and old bolt holes",
    "a glossy white lacquer table with rounded corners",
    "a dark walnut table with a live edge and visible cathedral grain",
    "a concrete-topped table with tiny air pockets and a sealed finish",
    "a navy-blue painted table with light distressing at the edges",
    "a bright red laminate table with a chrome banding strip",
    "a light gray linoleum-topped table with birch edging",
    "a green-tinted glass table over a steel frame",
    "a honey teak table with oiled finish and fine ribbon grain",
]
OBJECTS = [
    "a tall emerald-green anodized aluminum bottle with a matte finish and laser-etched logo",
    "a cobalt-blue glass bottle with a ribbed surface and cork stopper",
    "a matte charcoal thermos with a copper accent ring at the lid",
    "a white ceramic canister with thin cobalt pinstripes and a glossy glaze",
    "a brushed copper flask with a hammered texture and leather strap",
    "a translucent amber plastic bottle with a white flip-top cap",
    "a candy-red enameled steel bottle with tiny chips at the base rim",
    "a seafoam-green stoneware jar with visible throwing rings",
    "a bright yellow silicone-wrapped sports bottle with a gray nozzle",
    "a frosted lavender glass bottle with an embossed floral pattern",
    "a stainless bottle with a mirror polish and colorful vinyl stickers",
    "a deep teal ceramic bottle with a reactive drip glaze in cream",
    "a matte white bottle with sage-green typography and a bamboo cap",
    "an orange-and-cream striped tin canister with a slight sheen",
    "a graphite-gray insulated bottle with a knurled steel cap",
    "a pearl-white bottle with iridescent shimmer and rose-gold lid",
]
CONTAINERS = [
    "a natural wicker basket with a warm honey tone and visible weave",
    "a shallow matte-gray felt tray with stitched edges",
    "a pale pine crate with dovetail joints and a branded stamp",
    "a glossy white ceramic bowl with a wide flared rim",
    "a woven seagrass basket with a two-tone checker pattern",
    "a brushed aluminum tray with rolled edges and rivets",
    "a slate-blue stoneware dish with speckled glaze",
    "a clear acrylic bin with polished edges",
    "a forest-green plastic tote with molded handles",
    "a cork-lined walnut tray with rounded corners",
    "a galvanized steel bucket with a rustic zinc patina",
    "a terracotta bowl with an unglazed exterior and glazed interior",
]
LIGHTING = [
    "lit by warm afternoon sunlight from a side window",
    "under cool diffuse overhead studio lighting",
    "with soft morning light and long gentle shadows",
    "under bright neutral task lighting with crisp highlights",
    "in even overcast daylight with minimal shadows",
    "with warm tungsten accent lighting from the left",
]


def build_rich_prompt(seed: int) -> str:
    rng = random.Random(seed)
    return (
        f"This video shows in the background, a bright room with {rng.choice(WALLS)} "
        f"and {rng.choice(FLOORS)}, {rng.choice(LIGHTING)}. "
        f"In the foreground, on {rng.choice(TABLES)}, there is {rng.choice(OBJECTS)} "
        f"and {rng.choice(CONTAINERS)}. "
        f"The robot is instructed to pick up the bottle and place it into the container."
    )


if __name__ == "__main__":
    for i in range(3):
        print(f"[{i}] {build_rich_prompt(i)}\n")

"""Paper-style ImageNet text classifier inputs for CLIP-family models.

The class-name overrides and prompt templates follow the Apache-2.0 licensed
Google Big Vision zero-shot evaluator:
https://github.com/google-research/big_vision/tree/main/big_vision/evaluators/proj/image_text
"""

import re
import string
from typing import List


# Entries omitted here are identical to the canonicalized torchvision label.
_CLIP_CLASS_NAME_OVERRIDES = {
    4: "hammerhead shark",
    7: "rooster",
    15: "american robin",
    20: "american dipper",
    21: "kite bird of prey",
    25: "fire salamander",
    26: "smooth newt",
    27: "newt",
    30: "american bullfrog",
    33: "loggerhead sea turtle",
    34: "leatherback sea turtle",
    39: "green iguana",
    40: "carolina anole",
    41: "desert grassland whiptail lizard",
    43: "frillednecked lizard",
    46: "european green lizard",
    47: "chameleon",
    49: "nile crocodile",
    52: "worm snake",
    53: "ringnecked snake",
    54: "eastern hognosed snake",
    55: "smooth green snake",
    56: "kingsnake",
    62: "african rock python",
    66: "saharan horned viper",
    67: "eastern diamondback rattlesnake",
    68: "sidewinder rattlesnake",
    72: "yellow garden spider",
    74: "european garden spider",
    75: "southern black widow",
    83: "prairie grouse",
    84: "peafowl",
    87: "african grey parrot",
    97: "duck",
    121: "red king crab",
    132: "great egret",
    133: "bittern bird",
    136: "common gallinule",
    140: "dunlin",
    141: "common redshank",
    152: "japanese chin",
    153: "maltese",
    154: "pekingese",
    155: "shih tzu",
    156: "king charles spaniel",
    161: "basset hound",
    164: "bluetick coonhound",
    165: "black and tan coonhound",
    166: "treeing walker coonhound",
    168: "redbone coonhound",
    179: "staffordshire bull terrier",
    188: "wire fox terrier",
    191: "airedale terrier",
    192: "cairn terrier",
    194: "dandie dinmont terrier",
    195: "boston terrier",
    199: "scottish terrier",
    201: "australian silky terrier",
    204: "lhasa apso",
    215: "brittany dog",
    216: "clumber spaniel",
    217: "english springer spaniel",
    224: "groenendael dog",
    227: "australian kelpie",
    233: "bouvier des flandres dog",
    235: "german shepherd dog",
    236: "dobermann",
    240: "appenzeller sennenhund",
    241: "entlebucher sennenhund",
    243: "bullmastiff",
    247: "st bernard",
    248: "husky",
    249: "alaskan malamute",
    255: "leonberger",
    256: "newfoundland dog",
    257: "great pyrenees dog",
    260: "chow chow",
    262: "brussels griffon",
    263: "pembroke welsh corgi",
    264: "cardigan welsh corgi",
    268: "mexican hairless dog xoloitzcuintli",
    269: "grey wolf",
    270: "alaskan tundra wolf",
    271: "red wolf or maned wolf",
    275: "african wild dog",
    281: "tabby cat",
    285: "egyptian mau",
    296: "polar bear",
    303: "longhorn beetle",
    312: "cricket insect",
    313: "stick insect",
    315: "praying mantis",
    321: "red admiral butterfly",
    322: "ringlet butterfly",
    323: "monarch butterfly",
    324: "small white butterfly",
    326: "gossamerwinged butterfly",
    330: "cottontail rabbit",
    332: "angora rabbit",
    339: "common sorrel horse",
    341: "pig",
    348: "ram adult male sheep",
    349: "bighorn sheep",
    350: "alpine ibex",
    352: "impala antelope",
    358: "european polecat",
    371: "patas monkey",
    375: "blackandwhite colobus",
    378: "whiteheaded capuchin",
    380: "titi monkey",
    381: "geoffroys spider monkey",
    382: "common squirrel monkey",
    383: "ringtailed lemur",
    385: "asian elephant",
    386: "african bush elephant",
    387: "red panda",
    389: "snoek fish",
    391: "silver salmon",
    392: "rock beauty fish",
    393: "clownfish",
    395: "gar fish",
    397: "pufferfish",
    408: "amphibious vehicle",
    412: "trash can",
    418: "ballpoint pen",
    419: "bandaid",
    421: "baluster handrail",
    428: "wheelbarrow",
    433: "swimming cap",
    436: "station wagon",
    437: "lighthouse",
    439: "military hat bearskin or shako",
    442: "bell tower",
    443: "baby bib",
    444: "tandem bicycle",
    446: "ring binder",
    450: "bobsleigh",
    452: "poke bonnet",
    454: "bookstore",
    455: "bottle cap",
    456: "hunting bow",
    458: "brass memorial plaque",
    459: "bra",
    466: "highspeed train",
    468: "taxicab",
    469: "cauldron",
    477: "tool kit",
    478: "cardboard box carton",
    480: "automated teller machine",
    487: "mobile phone",
    491: "chainsaw",
    492: "storage chest",
    494: "bell or wind chime",
    498: "movie theater",
    502: "clogs",
    505: "coffeemaker",
    506: "spiral or coil",
    509: "candy store",
    517: "construction crane",
    520: "infant bed",
    528: "rotary dial telephone",
    533: "dishcloth",
    535: "disc brake",
    537: "dog sled",
    540: "drilling rig",
    550: "espresso machine",
    553: "filing cabinet",
    555: "fire truck",
    564: "fourposter bed",
    570: "gas mask or respirator",
    575: "golf cart",
    581: "radiator grille",
    584: "hair clip",
    586: "halftrack",
    589: "hair dryer",
    592: "hard disk drive",
    595: "combine harvester",
    601: "hoop skirt",
    602: "gymnastic horizontal bar",
    603: "horsedrawn vehicle",
    606: "clothes iron",
    607: "carved pumpkin",
    608: "jeans",
    610: "tshirt",
    612: "rickshaw",
    620: "laptop computer",
    628: "ocean liner",
    630: "slipon shoe",
    632: "music speaker",
    633: "loupe magnifying glass",
    634: "sawmill",
    636: "messenger bag",
    638: "tights",
    639: "onepiece bathing suit",
    648: "medicine cabinet",
    651: "microwave oven",
    661: "ford model t",
    666: "mortar and pestle",
    667: "graduation cap",
    670: "vespa",
    672: "tent",
    673: "computer mouse",
    677: "metal nail",
    680: "baby pacifier",
    681: "notebook computer",
    687: "pipe organ",
    690: "bullock cart",
    692: "product packet packaging",
    694: "paddle wheel",
    697: "pajamas",
    699: "pan flute",
    705: "railroad car",
    709: "pencil case",
    714: "plectrum",
    717: "pickup truck",
    724: "pirate ship",
    725: "drink pitcher",
    726: "block plane",
    730: "farm plow",
    737: "soda bottle",
    738: "plant pot",
    744: "missile",
    746: "hockey puck",
    751: "race car",
    758: "fishing casting reel",
    767: "eraser",
    769: "ruler measuring stick",
    770: "sneaker",
    773: "salt shaker",
    776: "saxophone",
    778: "weighing scale",
    782: "crt monitor",
    788: "shoe store",
    789: "shoji screen room divider",
    796: "balaclava ski mask",
    800: "slot machine",
    807: "solar thermal collector",
    810: "keyboard space bar",
    814: "motorboat",
    821: "through arch bridge",
    824: "scarf",
    829: "tram",
    831: "couch",
    836: "sunglasses",
    840: "mop",
    842: "swim trunks shorts",
    844: "electrical switch",
    850: "teddy bear",
    853: "thatched roof",
    854: "front curtain",
    856: "threshing machine",
    865: "toy store",
    867: "semitrailer truck",
    876: "hot tub",
    881: "upright piano",
    882: "vacuum cleaner",
    884: "vaulted or arched ceiling",
    885: "velvet fabric",
    895: "military aircraft",
    896: "sink",
    897: "washing machine",
    903: "hair wig",
    908: "airplane wing",
    912: "splitrail fence",
    913: "shipwreck",
    914: "sailboat",
    916: "website",
    918: "crossword",
    919: "traffic or street sign",
    921: "dust jacket",
    929: "popsicle",
    930: "baguette",
    934: "hot dog",
    935: "mashed potatoes",
    936: "cabbage",
    948: "granny smith apple",
    956: "cherimoya custard apple",
    960: "chocolate syrup",
    962: "meatloaf",
    964: "pot pie",
    968: "tea cup",
    970: "mountain",
    975: "lakeshore",
    978: "beach",
    981: "baseball player",
    982: "bridegroom",
    989: "rose hip",
    990: "horse chestnut seed",
    994: "stinkhorn mushroom",
    995: "earth star fungus",
    996: "hen of the woods mushroom",
    998: "corn cob",
    999: "toilet paper",
}


# OpenAI's tokenizer lowercases text, so only punctuation that is removed by
# Big Vision's canonicalization needs to be restored for exact paper prompts.
_OPENAI_CLIP_PUNCTUATED_CLASS_NAME_OVERRIDES = {
    21: "kite (bird of prey)",
    43: "frilled-necked lizard",
    53: "ring-necked snake",
    54: "eastern hog-nosed snake",
    89: "sulphur-crested cockatoo",
    98: "red-breasted merganser",
    202: "soft-coated wheaten terrier",
    205: "flat-coated retriever",
    206: "curly-coated retriever",
    247: "st. bernard",
    268: "mexican hairless dog (xoloitzcuintli)",
    326: "gossamer-winged butterfly",
    348: "ram (adult male sheep)",
    352: "impala (antelope)",
    359: "black-footed ferret",
    364: "three-toed sloth",
    375: "black-and-white colobus",
    378: "white-headed capuchin",
    381: "geoffroy's spider monkey",
    383: "ring-tailed lemur",
    419: "band-aid",
    421: "baluster / handrail",
    439: "military hat (bearskin or shako)",
    466: "high-speed train",
    478: "cardboard box / carton",
    489: "chain-link fence",
    564: "four-poster bed",
    573: "go-kart",
    586: "half-track",
    590: "hand-held computer",
    603: "horse-drawn vehicle",
    610: "t-shirt",
    630: "slip-on shoe",
    639: "one-piece bathing suit",
    692: "product packet / packaging",
    722: "ping-pong ball",
    739: "potter's wheel",
    789: "shoji screen / room divider",
    842: "swim trunks / shorts",
    867: "semi-trailer truck",
    912: "split-rail fence",
    956: "cherimoya (custard apple)",
    986: "yellow lady's slipper",
}


_CLIP_PAPER_PROMPT_TEMPLATES = (
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
    "{}",
)


def _canonicalize_text(text: str, *, keep_placeholder: bool = False) -> str:
    """Match Big Vision's lowercase and punctuation canonicalization."""

    text = str(text).replace("_", " ")
    if keep_placeholder:
        text = "{}".join(
            part.translate(str.maketrans("", "", string.punctuation))
            for part in text.split("{}")
        )
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text.lower()).strip()


def imagenet_clip_class_names() -> List[str]:
    """Return the paper evaluator's canonical class name at each ILSVRC index."""

    from torchvision.models import ResNet50_Weights

    names = [
        _canonicalize_text(name)
        for name in ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
    ]
    if len(names) != 1000:
        raise ValueError(f"expected 1000 torchvision ImageNet categories, got {len(names)}")
    for index, name in _CLIP_CLASS_NAME_OVERRIDES.items():
        names[index] = name
    return names


def clip_paper_prompt_templates() -> List[str]:
    """Return Big Vision's canonicalized `clip_paper` template ensemble."""

    templates = [
        _canonicalize_text(template, keep_placeholder=True)
        for template in _CLIP_PAPER_PROMPT_TEMPLATES
    ]
    if len(templates) != 81:
        raise ValueError(f"expected 81 CLIP paper prompt templates, got {len(templates)}")
    return templates


def imagenet_clip_prompt_groups() -> List[List[str]]:
    """Return 81 paper-style prompts for every ImageNet class."""

    templates = clip_paper_prompt_templates()
    return [
        [template.format(class_name) for template in templates]
        for class_name in imagenet_clip_class_names()
    ]


def openai_clip_class_names() -> List[str]:
    """Return the exact token-equivalent class names from OpenAI's notebook."""

    names = imagenet_clip_class_names()
    for index, name in _OPENAI_CLIP_PUNCTUATED_CLASS_NAME_OVERRIDES.items():
        names[index] = name
    return names


def openai_clip_prompt_templates() -> List[str]:
    """Return the 80 raw ImageNet templates used in the original CLIP paper."""

    templates = list(_CLIP_PAPER_PROMPT_TEMPLATES[:-1])
    if len(templates) != 80:
        raise ValueError(f"expected 80 OpenAI CLIP prompt templates, got {len(templates)}")
    return templates


def openai_clip_prompt_groups() -> List[List[str]]:
    """Return the original CLIP paper's 80 prompts for every ImageNet class."""

    templates = openai_clip_prompt_templates()
    return [
        [template.format(class_name) for template in templates]
        for class_name in openai_clip_class_names()
    ]

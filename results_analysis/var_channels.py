
LEVELS_6_STATS = {
    'global_means_path' : "/home/bingxing2/home/scx6115/Ocean_AI_model/sample_00/train/global_means.npy",
    'global_stds_path' :  "/home/bingxing2/home/scx6115/Ocean_AI_model/sample_00/train/global_stds.npy",
    'land_mask_file' : "/home/bingxing2/home/scx6115/Ocean_AI_model/sample_00/land_mask.h5",
}

VAR_INDEX_6_LEVELS = {
    "T0":0,  "T50":1,  "T100":2,  "T300":3,  "T500":4,  "T1000":5,
    "S0":6,  "S50":7,  "S100":8,  "S300":9,  "S500":10, "S1000":11,
    "U0":12, "U50":13, "U100":14, "U300":15, "U500":16, "U1000":17,
    "V0":18, "V50":19, "V100":20, "V300":21, "V500":22, "V1000":23,
    "SSH": 24,
} 

LEVELS_6 = [0, 50, 100, 300, 500, 1000]

LEVELS_15 = [0, 6, 10, 20, 30, 50, 70, 100, 125, 150, 200, 250, 300, 400, 500]

LEVELS_15_STATS = {
    'global_means_path' : "/home/bingxing2/home/scx6115/Ocean_AI_model/sample_01/train/global_means.npy",
    'global_stds_path' :  "/home/bingxing2/home/scx6115/Ocean_AI_model/sample_01/train/global_stds.npy",
    'land_mask_file' : "/home/bingxing2/home/scx6115/Ocean_AI_model/sample_01/land_mask.h5",
}

VAR_INDEX_15_LEVELS = {
        "T0":   0,   "T6":   1,   "T10":  2,   "T20":  3,   "T30":  4, 
        "T50":  5,   "T70":  6,   "T100": 7,   "T125": 8,   "T150": 9, 
        "T200": 10,  "T250": 11,  "T300": 12,  "T400": 13,  "T500": 14,
        "S0":   15,  "S6":   16,  "S10":  17,  "S20":  18,  "S30":  19, 
        "S50":  20,  "S70":  21,  "S100": 22,  "S125": 23,  "S150": 24, 
        "S200": 25,  "S250": 26,  "S300": 27,  "S400": 28,  "S500": 29,
        "U0":   30,  "U6":   31,  "U10":  32,  "U20":  33,  "U30":  34, 
        "U50":  35,  "U70":  36,  "U100": 37,  "U125": 38,  "U150": 39, 
        "U200": 40,  "U250": 41,  "U300": 42,  "U400": 43,  "U500": 44,
        "V0":   45,  "V6":   46,  "V10":  47,  "V20":  48,  "V30":  49, 
        "V50":  50,  "V70":  51,  "V100": 52,  "V125": 53,  "V150": 54, 
        "V200": 55,  "V250": 56,  "V300": 57,  "V400": 58,  "V500": 59,
        "SSH": 60,
} 

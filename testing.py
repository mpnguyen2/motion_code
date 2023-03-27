from preprocessing import load_UCR_UEA_data

names = [
'PenDigits',
'SmoothSubspace',
'Tiselac',
'MelbournePedestrian',
'ItalyPowerDemand',
'Chinatown',
'JapaneseVowels',
'RacketSports',
'InsectWingbeat',
'LSST',
'Libras',
'Crop',
'FingerMovements',
'NATOPS',
'SyntheticControl',
'SharePriceIncrease',
'FaceDetection',
'SonyAIBORobotSurface2',
'ERing',
'SonyAIBORobotSurface1',
'ProximalPhalanxTW',
'ProximalPhalanxOutlineCorrect',
'ProximalPhalanxOutlineAgeGroup',
'PhalangesOutlinesCorrect',
'MiddlePhalanxTW',
'MiddlePhalanxOutlineCorrect',
'MiddlePhalanxOutlineAgeGroup',
'DistalPhalanxTW',
'DistalPhalanxOutlineCorrect',
'DistalPhalanxOutlineAgeGroup',
'TwoLeadECG',
'MoteStrain',
'SpokenArabicDigits',
'ElectricDevices',
'ECG200',
'MedicalImages',
'BasicMotions' 
]

for name in names:
    num_sample, _ , Y, labels = load_UCR_UEA_data(name, mode='train', visualize=False)
    print(name, num_sample, Y.shape, labels.shape)
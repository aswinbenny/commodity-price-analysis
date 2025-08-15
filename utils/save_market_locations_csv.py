import pandas as pd

location_df = pd.DataFrame({'Market': ['KURUMASSERY VFPCK', 'Vengola  VFPCK', 'Keezhampara VFPCK', 'Edackattuvayal  VFPCK', 'Mazhuvannur VFPCK', 
                                       'Perumbavoor', 'Mookkannur  VFPCK', 'Thiruvaniyoor  VFPCK', 'POTHANIKKADU VFPCK', 'Nedungapra  VFPCK', 
                                       'THURAVOOR VFPCK', 'KANJOOR VFPCK', 'Koovapadi VFPCK', 'KARUMALOOR VFPCK', 'Moovattupuzha', 'Amalapuram  VFPCK',
                                       'Kothamangalam', 'Kunnukara  VFPCK', 'Malayattoor  VFPCK', 'Aluva', 'Ernakulam', 'Kadungallur  VFPCK', 
                                       'Puthenvelikkara  VFPCK', 'North Paravur', 'Angamaly', 'Broadway market', 'Piravam', 'Thrippunithura' ], 
                            'Latitude': [10.182338, 10.075988, 10.101587, 9.867798, 10.014496, 10.125262, 10.220011, 9.945238, 10.008046,
                                         10.125952, 10.202895, 10.136413, 10.160243, 10.128628, 9.984919, 10.226238, 10.062010, 10.160330,
                                         10.196592, 10.096921, 9.981858, 10.110571, 10.186876, 10.144213, 10.182566, 9.979023, 9.873363, 9.951943], 
                            'Longitude': [76.330828, 76.457453, 76.650773, 76.436062, 76.497585, 76.487684, 76.407823, 76.415859, 76.674538,
                                          76.550421, 76.424063, 76.425766, 76.478894, 76.292491, 76.580517, 76.460359, 76.627183,  76.295019,
                                          76.496859, 76.356250, 76.292906, 76.325008, 76.242306, 76.229465, 76.375519, 76.280133, 76.490172, 76.353458 ]})
location_df.to_csv('data/market_locations.csv', index=False)
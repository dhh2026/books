from functools import reduce
from typing import Callable

from hereutil import here, add_to_sys_path
from sqlframe_gizmosql import Window
add_to_sys_path(here())
from src.common_basis_gizmosql import *
subqueries: list[nw.LazyFrame] = []

import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from clean_helpers import *

data_fields = [#'all_titles', 
               'all_individual_actors', 
               'all_publishers', 
               'all_years_of_publication',
               # 'all_genre_terms',
               'all_primary_languages',
               'all_places_of_publication' 
               ]

sources = ['vd16', 'vd17', 'vd18']

# city for extraction
city = 'wittenberg'
city_names = [
            "Wittenberg",
            "Wittebergae",
            "Vitembergae",
            "Wittenbergae",
            "Witebergae",
            "[Wittenberg]",
            "Vitebergae",
            "Wittemberg",
            "Wittebergæ",
            "Vittembergae",
            "[Wittebergae]",
            "Vitembergæ",
            "Wittembergae",
            "Wittenbergk",
            "Wittenbergæ",
            "[Vitembergae]",
            "Vitembergae Saxonum",
            "[Wittenbergae]",
            "[Vitebergae]",
            "Witembergae",
            "VVittebergae",
            "Wjttenberg",
            "VVitebergae"
            ]


if __name__=='__main__':

    # query
    q = f(data_fields[0]).filter(c('source').is_in(sources), 
                                        ).drop(['field_number', 
                                                'field_code'
                                                ], strict=False)
    for field in data_fields[1:]:
        q = q.join(f(field)
                    .filter(c('source').is_in(sources))
                    .drop(['field_number',
                        'field_code'
                        ], strict=False),
                    how='left', on=['record_number', 'source']
                )
        
    # filter city: Wittenberg as example
    res = (q.filter(c('place_of_publication').is_in(city_names)))
    df = res.collect().to_native().to_pandas()

    # clean data and save
    df_cleaned = pd.DataFrame()
    df_cleaned['author'] = df.value.apply(lambda x: clean_author(str(x)) if pd.notna(x) else [])
    df_cleaned['publisher'] = df['publisher'].apply(lambda x: clean_text(x) if pd.notna(x) else '')
    df_cleaned['year'] = df['year_of_publication']
    df_cleaned['lang'] = df['primary_language_code']
    df_cleaned['rec_num'] = df['record_number']
    df_cleaned = df_cleaned[(df_cleaned['author']!='') & (df_cleaned['publisher']!='')].reset_index(drop=True)

    df_cleaned[['rec_num', 
            'author', 
            'publisher',
            'year',
            'lang',
            ]].to_json(f'../../data/work/vd_all_{city}.jsonl', lines=True, orient='records')
    
    # initate a json file to save the city network statistics
    time_edges = np.linspace(1500, 1800, 11)
    t_windows = []
    i = 0
    while i < len(time_edges) - 1: 
        start = time_edges[i]
        end = time_edges[i+1] -1
        t_windows.append((start, end))
        i += 1
    
    all_results = {'-'.join([str(int(t[0])), str(int(t[1]))]):{
                'status': None,
                'updeg_gini': None,
                'lowdeg_gini': None,
                'clustering': {'obs': None,
                                'coef_mean': None,
                                'coef_lb': None,
                                'coef_ub': None
                                },
                'distance': {'obs': None,
                                'coef_mean': None,
                                'coef_lb': None,
                                'coef_ub': None
                                }
                  } for t in t_windows}
    
    with open(f'../../data/work/{city}_stats.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    



#%%
from functools import reduce
from typing import Callable

from hereutil import here, add_to_sys_path
from sqlframe_gizmosql import Window
add_to_sys_path(here())
#from src.common_basis_local import *
#from src.common_basis_s3 import *
from src.common_basis_gizmosql import *
#from src.common_basis_quack import *

def gather_subfields(lnf: nw.LazyFrame[GizmoSQLDataFrame]) -> nw.LazyFrame[GizmoSQLDataFrame]:
    return to_narwhals(
        to_gizmosql(lnf)
            .groupBy(F.col('record_number'), F.col('field_number'), F.col('field_code'))
            .agg(F.array_join(F.transform(F.sort_array(F.collect_list(F.struct(F.col('subfield_number'), F.col('subfield_code'), F.col('value')))), lambda x: F.concat(x.subfield_code, F.lit('$'), x.value)), '|').alias("value"))
    )


#%%
def extract_title(dataset: str, standard: str):
    if standard in {'marc21', 'intermarc', 'danmarc'}:
       q =  (f(dataset)
            .filter(c('field_code') == '245', c('subfield_code') == 'a')
       )
    elif standard == 'unimarc':
         q = (
             f(dataset)
                .filter(c('field_code') == '200', c('subfield_code') == 'a')
         )
    elif standard == 'pica':
         q = (
             f(dataset)
                .filter(c('field_code') == '021A', c('subfield_code') == 'a')
         )
    elif standard == 'istc':
         q = (
             f(dataset)
                .filter(c('field_code') == 'title')
         )
    else:
        print(f"Skipping title extraction for dataset {dataset} with standard {standard}")
        return
    persist_as_s3_parquet("all_titles", dataset, q.select(c('record_number'), c('value').alias('main_title')))

iter_catalogues(extract_title)
register_s3_parquets_as_view("all_titles")

#%%
def extract_country_of_publication(dataset: str, standard: str):
    if standard == 'marc21':
        q = (f(dataset)
            .filter(c('field_code') == '008')
            .with_columns(value=c('value').str.slice(15, 3).str.strip_chars(' '))
        )
    elif standard == 'intermarc':
        q = (f(dataset)
            .filter(c('field_code') == '008')
            .with_columns(value=c('value').str.slice(29, 2))
        )
    elif standard == 'unimarc':
        q = (f(dataset)
            .filter(c('field_code') == '102', c('subfield_code') == 'a')
        )
    elif standard == 'pica':
        q = (f(dataset)
            .filter(c('field_code') == '019@', c('subfield_code') == 'a')
        )
    elif standard == 'danmarc':
        q = (f(dataset)
            .filter(c('field_code') == '008', c('subfield_code') == 'b')
        )
    elif standard == 'istc':
        q = (f(dataset)
            .filter(c('subfield_code') == 'geo_info_imprint_country_code')
        )
    else:
        print(f"Skipping country of publication extraction for dataset {dataset} with standard {standard}")
        return
    persist_as_s3_parquet("all_countries_of_publication", dataset, q
                        .filter(c('value')!='')
                        .select(c('record_number'), c('value').alias('country_of_publication')))
    
iter_catalogues(extract_country_of_publication)
register_s3_parquets_as_view("all_countries_of_publication")

#to_parquet("p_country_of_publication", nw.concat(subqueries).sort('e_id'))

#%%
def extract_year_of_publication(dataset: str, standard: str):
    if standard == 'marc21':
        q = (f(dataset)
            .filter(c('field_code') == '008')
            .with_columns(value=c('value').str.slice(7, 4))
        )
    elif standard == 'intermarc':
        q = (f(dataset)
            .filter(c('field_code') == '008')
            .with_columns(value=c('value').str.slice(8, 4))
        )
    elif standard == 'unimarc':
        q = (f(dataset)
            .filter(c('field_code') == '100')
            .with_columns(value=c('value').str.slice(9, 4))
        )
    elif standard == 'danmarc':
        q = (f(dataset)
            .filter(c('field_code') == '008', c('subfield_code') == 'a')
        )
    elif standard == 'pica':
        q = (f(dataset)
            .filter(c('field_code') == '011@')
        )
    elif standard == 'istc':
        q = (f(dataset)
            .filter(c('subfield_code') == 'date_of_item_single_date')
        )
    else:
        print(f"Skipping year of publication extraction for dataset {dataset} with standard {standard}")
        return
    persist_as_s3_parquet("all_years_of_publication", dataset, q
                        .filter(c('value').str.contains(r'^\d{4}$'))
                        .select(c('record_number'), c('value').alias('year_of_publication').cast(nw.Int32)))
    
iter_catalogues(extract_year_of_publication)
register_s3_parquets_as_view("all_years_of_publication")

#%%
def extract_primary_language(dataset: str, standard: str):
    if standard == 'marc21' and dataset != 'viaf' and f(dataset) is not None:
        q = (f(dataset)
            .filter(c('field_code') == '008')
            .with_columns(value=c('value').str.slice(35, 3))
        )
    elif standard == 'intermarc':
        q = (f(dataset)
            .filter(c('field_code') == '008')
            .with_columns(value=c('value').str.slice(31, 3))
        )
    elif standard == 'unimarc':
        q = (f(dataset)
            .filter(c('field_code') == '100')
            .with_columns(value=c('value').str.slice(22, 3))
        )
    elif standard == 'danmarc':
        q = (f(dataset)
            .filter(c('field_code') == '008', c('subfield_code') == 'a')
        )
    elif standard == 'pica' and f(dataset) is not None:
        q = (f(dataset)
            .filter(c('field_code') == '010@', c('subfield_code') == 'a')
        )
    else:
        print(f"Skipping primary language extraction for dataset {dataset} with standard {standard}")
        return
    persist_as_s3_parquet("all_primary_languages", dataset, q
                        .with_columns(value=c('value').str.strip_chars(' |'))
                        .filter(c('value')!='', c('value')!='und')
                        .select(c('record_number'), c('value').alias('primary_language_code')))
    
iter_catalogues(extract_primary_language)
register_s3_parquets_as_view("all_primary_languages")

# %%

def extract_cataloguing_date(dataset: str, standard: str):
    if standard in {'marc21', 'intermarc'}:
        q = to_narwhals(
            to_gizmosql(f(dataset).filter(c('field_code')=='008').select(c("record_number"),c("value").str.slice(0, 7).alias("date_created_raw")))
            .withColumn("date_created", F.expr("try_cast(try_strptime(date_created_raw, '%y%m%d') AS DATE)"))
    )
    elif standard == 'unimarc':
        q = to_narwhals(
            to_gizmosql(f(dataset).filter(c('field_code')=='100').select(c("record_number"), c("value").str.slice(0, 9).alias("date_created_raw")))
            .withColumn("date_created", F.expr("try_cast(try_strptime(date_created_raw, '%Y%m%d') AS DATE)"))
        )
    elif standard == 'danmarc':
        q = to_narwhals(
            to_gizmosql(f(dataset).filter(c('field_code')=='001', c('subfield_code') == 'd').select(c("record_number"), c("value").alias("date_created_raw")))
            .withColumn("date_created", F.expr("try_cast(try_strptime(date_created_raw, '%Y%m%d') AS DATE)"))
        )
    elif standard == 'pica':
        q = to_narwhals(
            to_gizmosql(f(dataset).filter(c('field_code')=='001A').select(c("record_number"), c("value").str.slice(6, 11).alias("date_created_raw")))
            .withColumn("date_created", F.expr("try_cast(try_strptime(date_created_raw, '%d-%m-%y') AS DATE)"))
        )
    else:
        print(f"Skipping cataloguing date extraction for dataset {dataset} with standard {standard}")
        return
    persist_as_s3_parquet("all_cataloguing_dates", dataset, q.select(c("record_number"), c("date_created"), c("date_created_raw")))
    
iter_catalogues(extract_cataloguing_date)
register_s3_parquets_as_view("all_cataloguing_dates")

# %%

def extract_last_modification_datetime(dataset: str, standard: str):
    if standard in {'marc21', 'unimarc'}:
        q = to_narwhals(
            to_gizmosql(f(dataset).filter(c('field_code')=='005').select(c("record_number"), c("value").alias("last_modification_datetime_raw")))
            .withColumn("last_modification_datetime", F.expr("try_strptime(last_modification_datetime_raw, '%Y%m%d%H%M%S.%f')"))
        )
    elif standard == 'danmarc':
        q = to_narwhals(
            to_gizmosql(f(dataset).filter(c('field_code')=='001', c('subfield_code') == 'c').select(c("record_number"), c("value").alias("last_modification_datetime_raw")))
            .withColumn("last_modification_datetime", F.expr("try_strptime(last_modification_datetime_raw, '%Y%m%d%H%M%S')"))
        )
    elif standard == 'pica':
        q = to_narwhals(
            to_gizmosql(
                f(dataset).filter(c('field_code')=='001B', c('subfield_code') == '0')
                .join(
                    f(dataset).filter(c('field_code')=='001B', c('subfield_code') == 't'),
                    on='record_number'
                )
                .select(c("record_number"), nw.concat_str(c("value").str.slice(6, 11), l('T'), c("value_right")).alias("last_modification_datetime_raw"))
            )
            .withColumn("last_modification_datetime", F.expr("try_strptime(last_modification_datetime_raw, '%Y%m%d%H%M%S.%f')"))
        )
    else:
        print(f"Skipping last modification datetime extraction for dataset {dataset} with standard {standard}")
        return
    persist_as_s3_parquet("all_last_modification_datetimes", dataset, q.select(c("record_number"), c("last_modification_datetime"), c("last_modification_datetime_raw")))

iter_catalogues(extract_last_modification_datetime)
register_s3_parquets_as_view("all_last_modification_datetimes")

#%%

def extract_place_of_publication(dataset: str, standard: str):
    if standard == 'pica':
        q = nw.concat([
            f(dataset).filter(c('field_code')=='033A', c('subfield_code')=='p').with_columns(place_type=l('unstandardized')), # place of publication
            f(dataset).filter(c('field_code')=='033D', c('subfield_code')=='p').with_columns(place_type=l('standardized')) # standardised location
        ])
    elif standard == 'marc21':
        q = nw.concat([
            f(dataset).filter(c('field_code')=='260', c('subfield_code')=='a').with_columns(place_type=l('unstandardized')), # place of publication
            f(dataset).filter(c('field_code')=='260', c('subfield_code')=='e').with_columns(place_type=l('unstandardized')), # place of manufacture 
            f(dataset).filter(c('field_code')=='264', c('subfield_code')=='a').with_columns(place_type=l('unstandardized')),
            f(dataset).filter(c('field_code')=='751', c('subfield_code')=='a').with_columns(place_type=l('standardized'))
        ])
    elif standard == 'istc':
        q = f(dataset).filter(c('subfield_code')=='imprint_place').with_columns(place_type=l('standardized'))
    else:
        print(f"Skipping publication place extraction for dataset {dataset} with standard {standard}")
        return
    persist_as_s3_parquet("all_places_of_publication", dataset, q.select(c("record_number"), c("value").alias("place_of_publication"), c("place_type")))

iter_catalogues(extract_place_of_publication)
register_s3_parquets_as_view("all_places_of_publication")

#%%

def extract_place_of_publication_id(dataset: str, standard: str):
    if standard == 'pica':
        q = f(dataset).filter(c('field_code')=='033D', c('subfield_code')=='7')
    elif standard == 'marc21':
        q = f(dataset).filter(c('field_code')=='751', c('subfield_code')=='0')
    elif standard == 'istc':
        q = f(dataset).filter(c('subfield_code')=='geo_info_geonames_id')
    else:
        print(f"Skipping publication place extraction for dataset {dataset} with standard {standard}")
        return
    persist_as_s3_parquet("all_place_of_publication_ids", dataset, q.select(c("record_number"), c("value").alias("place_of_publication_id")))

iter_catalogues(extract_place_of_publication_id)
register_s3_parquets_as_view("all_place_of_publication_ids")
#%%

def extract_publishers(dataset: str, standard: str):
    if standard == 'pica':
        q = f(dataset).filter(c('field_code')=='033A', c('subfield_code')=='n')
    elif standard == 'marc21':
        q = nw.concat([
            f(dataset).filter(c('field_code')=='260', c('subfield_code')=='b'), 
            f(dataset).filter(c('field_code')=='260', c('subfield_code')=='f'), # manufacturer
            f(dataset).filter(c('field_code')=='264', c('subfield_code')=='b'),
            f(dataset).filter(c('field_code')=='751', c('subfield_code')=='b')
        ])
    elif standard == 'istc':
        q = f(dataset).filter(c('subfield_code')=='imprint_place')
    else:
        print(f"Skipping publisher extraction for dataset {dataset} with standard {standard}")
        return
    persist_as_s3_parquet("all_publishers", dataset, q.select(c("record_number"), c("value").alias("publisher")))

iter_catalogues(extract_publishers)
register_s3_parquets_as_view("all_publishers")
# %%

def extract_individual_actors(dataset: str, standard: str):
    if standard == 'pica':
        q = gather_subfields(
                f(dataset).filter(c('field_code').is_in(['028A', '028B', '028C', '028G', '028Z']))
            )
    elif standard == 'marc21' and dataset!='dnb': # dnb is too large for the current server
        q = gather_subfields(
            f(dataset).filter(c('field_code').is_in(['100', '600', '700', '800']))
        )
    elif standard == 'istc':
        q = gather_subfields(
            f(dataset).filter(c('field_code')=='author')
        )
    else:
        print(f"Skipping actor extraction for dataset {dataset} with standard {standard}")
        return
    persist_as_s3_parquet("all_individual_actors", dataset, q.select(c("record_number"), c("field_number"), c("field_code"), c("value")))
    
iter_catalogues(extract_individual_actors)
register_s3_parquets_as_view("all_individual_actors")
# %%

def extract_corporate_actors(dataset: str, standard: str):
    if standard == 'pica':
        q = gather_subfields(
                f(dataset).filter(c('field_code').is_in(['029A', '029F', '029G']))
            )
    elif standard == 'marc21':
        q = gather_subfields(
            f(dataset).filter(c('field_code').is_in(['110', '610', '710', '810']))
        )
    else:
        print(f"Skipping actor extraction for dataset {dataset} with standard {standard}")
        return
    persist_as_s3_parquet("all_corporate_actors", dataset, q.select(c("record_number"), c("field_number"), c("field_code"), c("value")))
    
iter_catalogues(extract_corporate_actors)
register_s3_parquets_as_view("all_corporate_actors")
# %%
def extract_meeting_names(dataset: str, standard: str):
    if standard == 'marc21':
        q = gather_subfields(
            f(dataset).filter(c('field_code').is_in(['111', '611', '711', '811']))
        )
    else:
        print(f"Skipping meeting name extraction for dataset {dataset} with standard {standard}")
        return
    persist_as_s3_parquet("all_meeting_names", dataset, q.select(c("record_number"), c("field_number"), c("field_code"), c("value")))
    
iter_catalogues(extract_meeting_names)
register_s3_parquets_as_view("all_meeting_names")
# %%
def extract_genre_terms(dataset: str, standard: str):
    if standard == 'pica':
        q = (
                f(dataset).filter(c('field_code')=='044S', c('subfield_code')=='a')
            )
    elif standard == 'marc21':
        q = (
            f(dataset).filter(c('field_code').is_in(['655']), c('subfield_code')=='a')
        )
    else:
        print(f"Skipping genre term extraction for dataset {dataset} with standard {standard}")
        return
    persist_as_s3_parquet("all_genre_terms", dataset, q.select(c("record_number"), c("value").alias("genre_term")))

iter_catalogues(extract_genre_terms)
register_s3_parquets_as_view("all_genre_terms")

# %%

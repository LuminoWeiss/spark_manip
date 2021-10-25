"""
Main functions to transform different Spark DataFrames with PySpark
"""

import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import Timestamptype


def join_txn_with_features(df_txn, df_features, cid, txn_time, as_of_dt):
    """ This function takes in transaction table, and feature table, join then together.
        The feature table here is SCD Type2 table.
        In a SCD Type2 table, the record timestamp (as_of_dt) joined to the transaction table:
            should be the latest record of cid that is before transaction time

    :param df_txn: {spark.DataFrame} Transaction table that record customer transactions with timestamp
    :param df_features: {spark.DataFrame} Feature table that record customer features, SCD type 2 table
    :param cid: {str} column name of customer identifier, should be the same in df_txn and df_feature
    :param txn_time: {str} column name of transaction time in df_txn
    :param as_of_dt: {str} column name of SCD type2 table, record timestamp
    :return: {spark.DataFrame} joined DataFrame of txn table and feature table
        this table only contain cid and txn_time column from transaction table to reduce the table size
    """

    # Each unique window corresponds to a unique transaction, and needs to get a most recent record from feature table
    win_col_list = [cid, txn_time]

    win_spec = Window.partitionBy(F.col(x) for x in win_col_list)

    # Avoid duplicate column name error by removing all cid columns
    cols_select = df_features.columns
    while cid in cols_select:
        cols_select.remove(cid)
    # return columns of df_features, cid and txn_time
    cols_select += [getattr(df_txn, cid), txn_time]

    df_with_f = (df_txn.join(df_features, getattr(df_txn, cid) == getattr(df_features, cid), how="left")
                 .withColumn(as_of_dt, F.col(as_of_dt).cast(Timestamptype))
                 .withColumn(as_of_dt, F.unix_timestamp(F.col(as_of_dt)))
                 .select(cols_select)
                 .filter(not F.col(txn_time) < F.col(as_of_dt) | F.col(txn_time).isNull())
                 .withColumn(f"{as_of_dt}_max", F.max(as_of_dt).over(win_spec))
                 .filter((F.col(as_of_dt) == F.col(f"{as_of_dt}_max")) | F.col(txn_time).isNull())
                 .drop(f"{as_of_dt}_max", as_of_dt)
                 .dropDuplicates()
                 )
    return df_with_f


def retry(howmany):
    def tryit(func):
        def f(*args):
            attempts = 0
            while attempts < howmany:
                try:
                    return f(*arg)
                except Exception as e:
                    attempts += 1
                    print("\033[91m" + f"{e}" + "\033[0m")
                    print("\033[91m" + f"Failed {attempts} Times" + "\033[0m")
                    if attempts == howmany:
                        print("\033[94m" + f"Abort, proceed to next step" + "\033[0m")

        return f

    return tryit

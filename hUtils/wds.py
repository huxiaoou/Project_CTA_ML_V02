import numpy as np
import pandas as pd
import mysql.connector


class CDownloadEngineWDS(object):
    def __init__(self, host: str, user: str, passwd: str, database: str):
        self.host = host
        self.user = user
        self.passwd = passwd
        self.database = database

    def download_futures_positions_by_dates(
        self,
        bgn_date: np.int32,
        end_date: np.int32,
        download_values: list[str],
        futures_type: str,
    ) -> pd.DataFrame:
        """

        :param trade_date:
        :param download_values:
        :param futures_type: 'C' = commodity, 'E' =  equity index
        :return:
        """

        # init database
        pDataBase = mysql.connector.connect(
            host=self.host,
            user=self.user,
            passwd=self.passwd,
            database=self.database,  # port = 3306
        )
        cursor_object = pDataBase.cursor()

        # set query conditions
        wds_tab_name = {
            "C": "CCOMMODITYFUTURESPOSITIONS",
            "E": "CINDEXFUTURESPOSITIONS",
        }[futures_type]
        filter_for_dates = f"TRADE_DT >= '{bgn_date}' AND TRADE_DT <= '{end_date}'"
        download_values_list = ["TRADE_DT"] + download_values
        cmd_query = f"SELECT {', '.join(download_values_list)} FROM {wds_tab_name} WHERE {filter_for_dates}"

        # query
        cursor_object.execute(cmd_query)
        download_data = cursor_object.fetchall()
        df = pd.DataFrame(download_data, columns=download_values_list)
        df.sort_values(by=["TRADE_DT", "S_INFO_WINDCODE", "FS_INFO_TYPE", "FS_INFO_RANK"], inplace=True)

        # close
        pDataBase.close()
        return df

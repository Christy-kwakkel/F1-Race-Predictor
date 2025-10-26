import pandas as pd

def compute_features(df):
        # Form average finishing position of the current season, if there have been at least 3 races in the season
    # if less than 3 races, look at the last 3 races of a driver
    df = df.sort_values(by=["driverId", "year", "round"]).copy()

    df["driver_form"] = df.groupby("driverId")["positionOrder"]\
        .apply(lambda x: x.rolling(3, min_periods=1).mean()).reset_index(drop=True)
    df["constructor_form"] = df.groupby("constructorId")["positionOrder"]\
        .apply(lambda x: x.rolling(3, min_periods=1).mean()).reset_index(drop=True)

    df["driver_team_synergy"] = df["driver_form"] * 0.7 + df["constructor_form"] * 0.3
    df["is_post2022"] = (df["year"] >= 2022).astype(int)
    df["year_norm"] = df["year"] - df["year"].min()

    return df




    # driver_forms = []
    # for driver, group in df.groupby("driverId"):
    #     group = group.sort_values(by=["year", "round"])
    #     form_values = []

    #     for i, row in group.iterrows():
    #         # all races of the same season before the current race
    #         past_season = group[(group["year"] == row["year"]) & (group["round"] < row["round"])]
                                
    #         if len(past_season) >= 3:
    #             form = past_season["positionOrder"].mean()
    #         else:
    #             past_all = group[group["round"] < row["round"]]
    #             form = past_all["positionOrder"].tail(3).mean() if len(past_all) > 0 else None
            
    #         form_values.append(form)

    #     group["driver_form"] = form_values
    #     driver_forms.append(group)
        
    # df = pd.concat(driver_forms)

    # df["driver_form"] = df["driver_form"].fillna(df["positionOrder"].mean())


    # constructor_forms = []
    # for constructor, group in df.groupby("constructorId"):
    #     group = group.sort_values(by=["year", "round"])
    #     form_values = []

    #     for i, row in group.iterrows():
    #         past_season = group[(group["year"] == row["year"]) & (group["round"] < row["round"])]
    #         if len(past_season) >= 3:
    #             form = past_season["positionOrder"].mean()
    #         else:
    #             past_all = group[group["round"] < row["round"]]
    #             form = past_all["positionOrder"].tail(3).mean() if len(past_all) > 0 else None

    #         form_values.append(form)

    #     group["constructor_form"] = form_values
    #     constructor_forms.append(group)

    # df = pd.concat(constructor_forms)
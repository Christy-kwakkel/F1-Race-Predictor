import pandas as pd

def compute_features(df, exclude_current_race=False, only_past_data=False):
    df = df.sort_values(by=["driverId", "year", "round"])

    # Form average finishing position of the current season, if there have been at least 3 races in the season

    # if less than 3 races, look at the last 3 races of a driver

    driver_forms = []
    for driver, group in df.groupby("driverId"):
        group = group.sort_values(by=["year", "round"])
        form_values = []

        for i, row in group.iterrows():
            # all races of the same season before the current race
            past_season = group[(group["year"] == row["year"]) & (group["round"] < row["round"])]
                                
            if len(past_season) >= 3:
                form = past_season["positionOrder"].mean()
            else:
                past_all = group[group["round"] < row["round"]]
                form = past_all["positionOrder"].tail(3).mean() if len(past_all) > 0 else None
            
            form_values.append(form)

        group["driver_form"] = form_values
        driver_forms.append(group)
        
    df = pd.concat(driver_forms)

    df["driver_form"] = df["driver_form"].fillna(df["positionOrder"].mean())


    constructor_forms = []
    for constructor, group in df.groupby("constructorId"):
        group = group.sort_values(by=["year", "round"])
        form_values = []

        for i, row in group.iterrows():
            past_season = group[(group["year"] == row["year"]) & (group["round"] < row["round"])]
            if len(past_season) >= 3:
                form = past_season["positionOrder"].mean()
            else:
                past_all = group[group["round"] < row["round"]]
                form = past_all["positionOrder"].tail(3).mean() if len(past_all) > 0 else None

            form_values.append(form)

        group["constructor_form"] = form_values
        constructor_forms.append(group)

    df = pd.concat(constructor_forms)

    # 

    return df
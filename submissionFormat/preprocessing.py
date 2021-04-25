import pandas as pd 

#read data from csv file into a pandas dataframe
with open(".//ipl_csv2//all_matches.csv") as f:
    ipl_data=pd.read_csv(f)

# print all columns
# print (data.columns)
# all columns
#   ['match_id','season','start_date','venue','innings','ball',
#   'batting_team','bowling_team','striker','non_striker','bowler',
#   'runs_off_bat','extras','wides','noballs','byes','legbyes',
#   'penalty','wicket_type','player_dismissed','other_wicket_type','other_player_dismissed']

    relevantColumns=['match_id','venue','innings','ball',
                'batting_team','bowling_team','striker','non_striker','bowler',
                'runs_off_bat','extras','wides','noballs','byes','legbyes',
                'penalty']
    ipl_data=ipl_data[relevantColumns]

# create another column that tells the number of runs scored, including off the bat and
# extra runs conceded by bowling team
    ipl_data['total_runs']=ipl_data['runs_off_bat'+ipl_data['extras']
                                    
# dropping columns with no further use
    ipl_data=ipl_data.drop(columns=['runs_off_bat','extras'])

# select only rows belonging to first 6 overs
    ipl_data=ipl_data[ipl_data['ball']<=5.6]

    ipl_data=ipl_data[ipl_data['innings']<=2]

# preprocess the data sp that we get a tuple of the following kind in each row
#   {['match_id','venue','innings','batting_team','bowling_team'],total_runs'}

    ipl_data=ipl_data.groupby(['match_id',
                            'venue',
                            'innings',
                            'batting_team',
                            'bowling_team']).total_runs.sum()

# convert back to dataframe
    ipl_data=ipl_data.reset_index()
    ipl_data=ipl_data.drop(columns=['match_id'])
    ipl_data.to_csv("myPreprocessed.csv",index=False)
#print('It's type is ',ipl_data.columns)

# data_scraper.py
import pandas as pd
import random
import time
import os
import logging


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def scrape_nba_data(seasons, teams, output_path):
    """
    Scrape NBA game logs from basketball-reference.com
    
    Parameters:
    seasons (list): List of seasons to scrape (e.g., ['2019', '2020'])
    teams (list): List of team abbreviations (e.g., ['atl', 'bos'])
    output_path (str): Path to save the CSV file
    
    Returns:
    pandas.DataFrame: Combined dataframe with all game logs
    """
    # Stats to extract and rename
    stats = [
        'FG', 'FGA', 'FG%', '3P', '3PA', '3P%',
        'FT', 'FTA', "FT%", 'ORB', 'TRB', 'AST',
        'STL', 'BLK', 'TOV', 'PF'
    ]
    
    # Stats dictionary for renaming
    tm_stats_dict = {stat: 'Tm_' + str(stat) for stat in stats}
    opp_stats_dict = {'Opp_' + str(stat) + '.1': 'Opp_' + str(stat) for stat in stats}
    
    nba_df = pd.DataFrame()
    
    for season in seasons:
        logger.info(f"Scraping data for season {season}")
        for team in teams:
            url = f'https://www.basketball-reference.com/teams/{team}/{season}/gamelog/'
            logger.info(f"Scraping: {url}")
            
            try:
                # Retrieving game stats from table 
                team_df = pd.read_html(url, header=1, attrs={'id': 'team_game_log_reg'})[0]
                
                # Drop empty row
                team_df = team_df[(team_df['Rk'].str != '') & (team_df['Rk'].str.isnumeric())]
                
                # Rename columns
                team_df = team_df.rename(columns={'Unnamed: 3': 'Home', 'Tm': 'Tm_Points', 'Opp.1': 'Opp_Points'})
                team_df = team_df.rename(columns=tm_stats_dict)
                team_df = team_df.rename(columns=opp_stats_dict)
                
                # Convert 'Home' column (@ = away game, empty = home game)
                team_df['Home'] = team_df['Home'].apply(lambda x: 0 if x == '@' else 1)
                
                # Add season and team columns
                team_df.insert(loc=0, column='Season', value=int(season))  # Convert to integer
                team_df.insert(loc=1, column='Team', value=team.upper())
                
                # Append to main dataframe
                nba_df = pd.concat([nba_df, team_df], ignore_index=True)
                
                # Pause to abide by website scraping restrictions
                time.sleep(random.randint(4, 6))
                
            except Exception as e:
                logger.error(f"Error scraping {team} {season}: {str(e)}")
    
    # Save the dataframe to CSV
    logger.info(f"Saving data to {output_path}")
    nba_df.to_csv(output_path, index=False)
    
    return nba_df
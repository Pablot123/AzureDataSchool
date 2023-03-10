
from funcs.myfuncs import cln_data

experiment_name ='exp-prueba'
mlflow.set_experiment(experiment_name)

with mlflow.start_run() as run:
    
    #loading data 
    #dataset = Dataset.get_by_name(ws, name='AirlinesDelay')
    df = pd.read_csv("../airlines_delay/airlines_delay.csv", sep = ",")
    
    #Distribution of the target column
    class_cero, class_one = df['Class'].value_counts()
    mlflow.log_metrics({'one':class_one, 'cero':class_cero})
    
    #Comparation of the mean
    mean_comparison = df[['Time','Length', 'Class']].groupby('Class').mean().to_dict()
 
    mlflow.log_metrics({'Mean Time class cero':mean_comparison['Time'][0], 'Mean Time class one':mean_comparison['Time'][1]})
    mlflow.log_metrics({'Mean Lenght class cero':mean_comparison['Length'][0], 'Mean Length class one':mean_comparison['Length'][1]})

    #Number of airlines
    number_of_airlines = len(df['Airline'].unique())

    mlflow.log_metric('number of airlines', number_of_airlines)

    #flights per airlnes
    flights_per_airline = df['Airline'].value_counts().to_dict()
    mlflow.log_metrics(flights_per_airline)

    
    #airports
    number_of_airports = len(df['AirportFrom'].unique())
    mlflow.log_metric('number of airports', number_of_airports)

    #airports with more activity
    arports_activity_df = df['AirportFrom'].value_counts().sort_values(ascending=False) + df['AirportTo'].value_counts().sort_values(ascending=False)
    top_5_airpots_activity = arports_activity_df.sort_values(ascending=False).head(5).to_dict()

    top_and_lowest_ariports_activity = {}
    for k, _ in top_5_airpots_activity.items():
        top_and_lowest_ariports_activity[f'Airport {k}'] = top_5_airpots_activity[k]


    #airports with more activity
    lowest_5_airpots_activity = arports_activity_df.sort_values(ascending=False).tail(5).to_dict()
    for k, _ in lowest_5_airpots_activity.items():
        top_and_lowest_ariports_activity[f'Airport {k}'] = lowest_5_airpots_activity[k]

    mlflow.log_metrics(top_and_lowest_ariports_activity)

    #DaysWeek
    dict_day_of_week= {1:'Monday', 2:'Tuesday', 3:'Wednesday', 4:'Thursday', 5:'Friday', 6:'Saturday', 7: 'Sunday'}
    df_day_of_week = df.replace({"DayOfWeek": dict_day_of_week},inplace=False)
    fligths_per_day_of_week = df_day_of_week['DayOfWeek'].value_counts().to_dict()

    mlflow.log_metrics(fligths_per_day_of_week)


    #plot distribution of time
    fig = plt.figure(figsize=(6,6))
    ax = fig.gca()    
    plt.hist(df['Length'])
    plt.show()
    mlflow.log_figure(fig, "plot.png")
    print("Plot listo!")
    

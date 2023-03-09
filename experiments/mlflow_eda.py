
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
    cls_one_df = df.loc[df['Class']==1]
    mean_one = cls_one_df['Time'].mean()
    
    cls_cero_df = df.loc[df['Class']==0]
    mean_cero = cls_cero_df['Time'].mean()
    
    mlflow.log_metrics({'mean cero':mean_cero, 'mean one':mean_one})

    #plot distribution of time
    fig = plt.figure(figsize=(6,6))
    ax = fig.gca()    
    sns.displot(df['Length'])
    plt.show()
    mlflow.log_figure(fig, "plot.png")
    print("Plot listo!")
    
    
    #fig = plt.bar(x=['0','1'], height=[class_cero, class_one])
    #mlflow.log_figure(fig, 'plot.png')
    
    

import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def generateHistorgrams(df: pd.DataFrame):
    # Age distribution
    plt.figure()
    sns.histplot(dataf['Age at enrollment'])
    plt.title("Wiek podczas zapisania się na studia")
    plt.ylabel("Liczba studentów")
    plt.xlabel("Wiek")
    plt.savefig('.\\results\\visualizations\\age_distribution.png')

    # Average grade at 1st semester distribution
    plt.figure(figsize=(12, 8))
    sns.histplot(data=df, x='Curricular units 1st sem (grade)', bins=20)
    plt.title('Rozkład ocen w pierwszym semestrze')
    plt.xlabel("Srednia ocen")
    plt.ylabel("Liczba studentów")
    plt.savefig('.\\results\\visualizations\\grade1_distribution.png')

    # Average grade at 2nd semester distribution
    plt.figure(figsize=(12, 8))
    sns.histplot(data=df, x='Curricular units 2nd sem (grade)', bins=20)
    plt.title('Rozkład ocen w drugim semestrze')
    plt.xlabel("Srednia ocen")
    plt.ylabel("Liczba studentów")
    plt.savefig('.\\results\\visualizations\\grade2_distribution.png')
    


def generateBoxplots(df: pd.DataFrame):
    # Age by target
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Target', y='Age at enrollment', data=df)
    plt.title('Wynik studiów względem wieku')
    plt.xlabel("Wynik studiów")
    plt.ylabel("Wiek")
    plt.savefig('.\\results\\visualizations\\boxplot_age_by_target.png')

    

    # Grade in 1st semester by course
    top_courses = df['Course'].value_counts().nlargest(5).index
    df_top_courses = df[df['Course'].isin(top_courses)]
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Course', y='Curricular units 1st sem (grade)', data=df_top_courses)
    plt.title('Rozkład ocen w pierwszym semestrze dla najpopularniejszych kursów')
    plt.xlabel("Kurs")
    plt.ylabel("Średnia ocen")
    # plt.xticks(rotation=45)
    plt.savefig('.\\results\\visualizations\\boxplot_grades_by_course.png')

def generateViolinplots(df):
    
    # Age by marital status
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Marital status', y='Age at enrollment', data=df)
    plt.title('Wiek a status matrymonialny')
    plt.ylabel("Wiek")
    plt.xlabel("Status matrymonialny")
    plt.savefig('.\\results\\visualizations\\violinplot_age_by_marital.png')
    
    # Grade in 1st semester by target
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Target', y='Curricular units 1st sem (grade)', data=df)
    plt.title('Rozkład ocen w 1. semestrze vs wynik studiów')
    plt.xlabel('Wynik studiów')
    plt.ylabel('Średnia ocen (1. semestr)')
    plt.savefig('.\\results\\visualizations\\grades1_vs_target.png')

     # Grade in 2nd semester by target
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Target', y='Curricular units 2nd sem (grade)', data=df)
    plt.title('Rozkład ocen w 2. semestrze vs wynik studiów')
    plt.xlabel('Wynik studiów')
    plt.ylabel('Średnia ocen (2. semestr)')
    plt.savefig('.\\results\\visualizations\\grades2_vs_target.png')

def generateErrorBars(df):

    # Średnie oceny z przedziałami ufności względem grupy wiekowej
    plt.figure(figsize=(12, 8))
    sns.pointplot(x='Age group', y='Curricular units 1st sem (grade)', 
                  data=df, capsize=0.1,order=["<20","20-25","25-30","30-40",">40"])
    plt.title('Średnie oceny w pierwszym semestrze z przedziałami ufności 95%')
    plt.xlabel("Grupa wiekowa")
    plt.ylabel("Średnia ocen")
    plt.savefig('.\\results\\visualizations\\errorbar_grades_by_age.png')
    
    # Średni wiek z przedziałami ufności względem Target
    plt.figure(figsize=(12, 8))
    sns.pointplot(x='Target', y='Age at enrollment', 
                 data=df, capsize=0.1)
    plt.title('Średni wiek z przedziałami ufności 95% względem wyniku studiów')
    plt.xlabel("Wynik studiów")
    plt.ylabel("Wiek")
    plt.savefig('.\\results\\visualizations\\errorbar_age_by_target.png')
    
    

def generateConditionalHistograms(df):
    top_dropout_courses = df[df['Target'] == 'Dropout']['Course'].value_counts().nlargest(5).index

    # Filtracja danych
    plot_data = df[df['Course'].isin(top_dropout_courses)]
    plt.figure(figsize=(14, 12))
    sns.histplot(
        data=plot_data,
        x='Course',
        hue='Target',
        multiple='stack',
        palette={"Graduate": "green", "Dropout": "red","Enrolled":"blue"},
        edgecolor='white',
        linewidth=0.5
    )
    plt.subplots_adjust(bottom=0.3)
    plt.title('Rozkład wyników studiów w top 5 kursach z największą liczbą porzuceń')
    plt.ylabel('Liczba studentów')
    plt.xlabel('Kurs')
    plt.xticks(rotation=-45)
    plt.savefig('.\\results\\visualizations\\conditional_histogram_dropout_courses.png')
    
    
def generateHeatmap(df):
    # Wybór kolumn numerycznych do korelacji
    numerical_cols = ["Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP"]
    
    # Obliczenie macierzy korelacji
    corr = df[numerical_cols].corr()

    # Heatmapa korelacji z optymalizacją etykiet
    plt.figure(figsize=(15, 15))
    heatmap = sns.heatmap(
        corr,
        cmap='coolwarm', 
        cbar_kws={"shrink": 0.8},  # Zmniejszenie kolorbaru
        linewidths=0.5,  # Cienkie linie między komórkami
        square=True  # Zachowanie proporcji kwadratowych
    )

    # Dostosowanie etykiet
    heatmap.set_xticklabels(
        heatmap.get_xticklabels(),
        rotation=45,
        ha='right',
        fontsize=10
    )
    heatmap.set_yticklabels(
        heatmap.get_yticklabels(),
        rotation=0,
        fontsize=10
    )

    plt.title('Macierz korelacji dla cech numerycznych', pad=20, fontsize=14)
    plt.tight_layout()  # Automatyczne dopasowanie
    plt.savefig('.\\results\\visualizations\\heatmap_correlation.png', dpi=100, bbox_inches='tight')
    

    plt.figure(figsize=(10, 8))
    # Obliczamy procenty w każdej grupie
    cross_tab = pd.crosstab(df['Marital status'], df['Target'], normalize='index') * 100
    sns.heatmap(cross_tab, annot=True, fmt='.1f', cmap='Blues', cbar_kws={'label': '% studentów'})
    plt.title('Rozkład wyników studiów wg statusu matrymonialnego (%)')
    plt.xlabel('Wynik studiów')
    plt.ylabel('Status matrymonialny')
    plt.tight_layout()
    plt.savefig('.\\results\\visualizations\\marital_status_vs_target_heatmap.png')


    

def analyzeCategoricalFeatures(df):
    # Analiza cech kategorialnych - częstości występowania kategorii
    categorical_cols = ['Marital status', 'Gender', 'Target']
    
    for col in categorical_cols:
        plt.figure(figsize=(12, 8))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index)
        plt.title(f'Rozkład dla cechy: {col}')
        plt.tight_layout()
        plt.savefig(f'.\\results\\visualizations\\countplot_{col.lower().replace(" ", "_")}.png')
        

if __name__ == "__main__":
    if (os.path.exists(".\\data\\datasetClean.csv")):
        dataf: pd.DataFrame = pd.read_csv(".\\data\\datasetClean.csv")
    else :
        print("First run loadData.py to load the dataset.")
        exit()

    if (not os.path.exists(".\\results\\visualizations")):
        os.makedirs(".\\results\\visualizations")
    
    sns.set_theme(style="whitegrid",font_scale=1.5)
    plt.rcParams['figure.figsize'] = (12, 12)


    generateHistorgrams(dataf)
    generateBoxplots(dataf)
    generateViolinplots(dataf)
    generateErrorBars(dataf)
    generateConditionalHistograms(dataf)
    generateHeatmap(dataf)
    analyzeCategoricalFeatures(dataf)
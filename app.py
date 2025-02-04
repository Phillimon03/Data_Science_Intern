import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageDraw
import base64
from io import BytesIO
from datetime import datetime


#Load the logo image
logo_path = r"C:\Users\PC\Desktop\FB_IMG_17285456011763622.jpg"
logo = Image.open(logo_path)

logo = logo.resize((100, 100))
bigsize = (logo.size[0] * 3, logo.size[1] * 3)
mask = Image.new('L', bigsize, 0)
draw = ImageDraw.Draw(mask)
draw.ellipse((0, 0) + bigsize, fill=255)
mask = mask.resize(logo.size)
logo.putalpha(mask)

buffered = BytesIO()
logo.save(buffered, format="PNG")
logo_base64 = base64.b64encode(buffered.getvalue()).decode()

#Set page config with logo
st.set_page_config(
    page_title="Corporate 24 Healthcare Dashboard",
    layout="wide",
    page_icon="data:image/png;base64," + logo_base64
)


st.sidebar.markdown("<h1 style='text-align: center; color: blue; font-size: 40px;'><b>DASHBOARD</b></span>", unsafe_allow_html=True)

def convert_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

st.markdown('<div style="position: fixed; bottom: 0; width: 100%; text-align: center;"><p><a href="https://www.google.com/url?client=internal-element-cse&cx=e644b33a7719ef93e&q=https://corp24med.com/&sa=U&ved=2ahUKEwiY-9KogoGKAxWXVKQEHYNyC64QFnoECAsQAQ&usg=AOvVaw2wI4TeGHO7OmicfGawpMoe&fexp=72821502,72821501">For More Info, Click here to Visit our Website</a></p></div>', unsafe_allow_html=True)

# Convert logo to base64
logo_base64 = convert_image_to_base64(logo)

# Set sidebar background color and center alignment
st.sidebar.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: blue;  /* Change this to your desired blue color */
        text-align: center;          /* Center align all items */
    }
    .upload-button {
        background-color: #d3d3d3;  /* Grey background */
        color: blue;                /* Blue text */
        border: none;               /* No border */
        padding: 10px 20px;        /* Padding */
        border-radius: 5px;        /* Rounded corners */
        font-size: 16px;           /* Font size */
        cursor: pointer;            /* Pointer cursor on hover */
        display: inline-block;      /* Inline block for centering */
        margin: 20px auto;         /* Center the button */
        width: 80%;                 /* Width for buttons */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar with round logo
st.sidebar.markdown(
    f"""
    <div style="width: 150px; height: 150px; border-radius: 150px; background-color: white; display: flex; justify-content: center; align-items: center; margin: 0 auto;">
        <img src="data:image/png;base64,{logo_base64}" style="border-radius: 150px; width: 100%; height: 100%; object-fit: cover;">
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("<h1 style='text-align: center; color: red; font-size: 24px;'><b>CORPORATE 24 HEALTHCARE</b></span>", unsafe_allow_html=True)

st.sidebar.markdown("<h1 style='text-align: center; color: green; font-size: 20px;'><b>Upload Dataset (.CSV)</b></span>", unsafe_allow_html=True)

# Initialize a variable for the dataset
df = None

# Handle file upload
uploaded_file = st.sidebar.file_uploader("Upload Your Dataset", type=["csv"], key="file_uploader", label_visibility="collapsed")

# Check if a file is uploaded
if uploaded_file is not None:
    try:
        # Attempt to read the dataset
        df = pd.read_csv(uploaded_file)

        # Proceed with further processing only if the dataframe is not empty
        if df.empty:
            st.warning("The uploaded file is empty. Please upload a valid CSV file.")
        else:
            # Convert 'DATE OF BIRTH' and 'VISIT DATE' to datetime
            df['DATE OF BIRTH'] = pd.to_datetime(df['DATE OF BIRTH'], errors='coerce')
            df['VISIT DATE'] = pd.to_datetime(df['VISIT DATE'], errors='coerce')

            # Extract age for filtering
            df['AGE'] = (datetime.now() - df['DATE OF BIRTH']).dt.days // 365
            # Centered title for the dataset on the main page
            st.markdown("<h2 style='text-align: center; color: purple'>PATIENTS DATASET</h2>", unsafe_allow_html=True)
            
            # Display the dataset filling the whole main page
            st.dataframe(df, use_container_width=True)  # Use container width to fill the page

            # Sidebar filters
            st.sidebar.markdown("<h2 style='text-align: center;'>FILTER OPTIONS</h2>", unsafe_allow_html=True)

            # Filter by SEX
            sex_filter = st.sidebar.selectbox("Filter by SEX:", options=["All", "Male", "Female"])
            if sex_filter != "All":
                df = df[df['SEX'] == sex_filter]

            # Filter by AGE
            age_filter = st.sidebar.slider("Filter by AGE:", min_value=0, max_value=100, value=(0, 100))
            df = df[(df['AGE'] >= age_filter[0]) & (df['AGE'] <= age_filter[1])]

            # Filter by MEDICAL AID
            medical_aid_filter = st.sidebar.multiselect("Filter by MEDICAL AID:", options=df['MEDICAL AID'].unique())
            if medical_aid_filter:
                df = df[df['MEDICAL AID'].isin(medical_aid_filter)]

            # Display the filtered dataset on the main page
            if not df.empty:
                # Title for filtered output
                st.markdown("<h2 style='text-align: center; color: purple'>FILTERED DATA</h2>", unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True)  # Display the filtered dataset

                # Save filtered dataset to CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Filtered Dataset as CSV",
                    data=csv,
                    file_name='filtered_dataset.csv',
                    mime='text/csv'
                )

                # Printing functionality
                if st.button("Print Filtered Dataset"):
                    # Create a temporary HTML file to print
                    html = df.to_html(index=False)
                    # Use st.markdown to create a button that opens a new tab
                    st.markdown(f'<a href="data:text/html;charset=utf-8,{html}" target="_blank" download="filtered_dataset.html">Click here to print the filtered dataset</a>', unsafe_allow_html=True)
            else:
                st.warning("No patients found matching your filters.")
            
            # Button for visualization
            if st.sidebar.markdown("<div style='text-align: center;'><button style='background-color: blue; color: white; padding: 10px 20px; border: none; boarder-radius: 5px; cursor: pointer;' type='button' onclick='this.style.background=\"green\"'>VISUALIZE THE DATA</button></div>", unsafe_allow_html=True):
                st.session_state.visualize = True
            
                st.markdown('<hr style="border: 2px solid blue;">', unsafe_allow_html=True)
                

                if st.markdown("<div style='text-align: center;'><button style='background-color: black; color: white; padding: 1px 400px; border: none; boarder-radius: 5px; cursor: pointer;' type='button' onclick='this.style.background=\"green\"'><u><b>TOTAL VISITS PER MONTH FOR EACH YEAR</b></u></button></div>", unsafe_allow_html=True):
                    st.markdown("<h2 style='text-align: center;'></h2>", unsafe_allow_html=True)
                    
                    #Step 2: Ensure 'VISIT DATE' is in datetime format
                    if 'VISIT DATE' not in df.columns:
                        st.error("Error: 'VISIT DATE' column not found in the dataset.")
                    else:
                        df['VISIT DATE'] = pd.to_datetime(df['VISIT DATE'], errors='coerce')
                        if df['VISIT DATE'].isna().all():
                            st.error("Error: All values in 'VISIT DATE' could not be converted to datetime.")
                        else:
                            # Step 3: Extract year and month from 'VISIT DATE'
                            df['Year'] = df['VISIT DATE'].dt.year
                            df['Month'] = df['VISIT DATE'].dt.month

                            # Step 4: Generate visualizations for all years in a single figure
                            years = [2020, 2021, 2022, 2023, 2024]  # List of years to filter and plot

                            fig, axes = plt.subplots(nrows=1, ncols=len(years), figsize=(20, 6), sharey=True)

                            for i, year in enumerate(years):
                                # Filter data for the specific year
                                year_data = df[df['Year'] == year]

                                # Aggregate visits by month
                                monthly_visits = year_data.groupby('Month').size().reset_index(name='Total Visits')

                                # Plot the data for the year in a specific subplot
                                ax = axes[i]
                                ax.bar(monthly_visits['Month'], monthly_visits['Total Visits'], color='red')

                                # Add labels and title
                                ax.set_title(f'{year}', fontsize=14)
                                ax.set_xlabel('Month', fontsize=12)
                                if i == 0:
                                    ax.set_ylabel('Total Visits', fontsize=12)
                                ax.set_xticks(range(1, 13))
                                ax.set_xticklabels([
                                'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
                                    ])
                                ax.grid(axis='y', linestyle='--', alpha=0.7)

                            # Adjust layout to prevent overlap
                            plt.tight_layout()
                            plt.suptitle('Total Visits Per Month for Each Year', fontsize=16, y=1.02)
                            plt.subplots_adjust(top=0.85)

                            # Display the figure
                            st.pyplot(fig)

                    #Group data by 'Year' and 'Month', and calculate the total number of visits
                    trend_data = df.groupby(['Year', 'Month']).size().reset_index(name='Total Visits')


                    #Create a line plot
                    fig, ax = plt.subplots(figsize=(12, 4))

                    #Plot data for each year
                    for year in trend_data['Year'].unique():
                        year_data = trend_data[trend_data['Year'] == year]
                        ax.plot(year_data['Month'], year_data['Total Visits'], label=f'{year}')

                    #Set title and labels
                    ax.set_title('Monthly Visits Trend from 2020 to 2024')
                    ax.set_xlabel('Month')
                    ax.set_ylabel('Total Visits')

                    #Set x-axis ticks and labels
                    ax.set_xticks(range(1, 13))
                    ax.set_xticklabels([
                        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
                        ])

                    #Add legend
                    ax.legend()

                    #Display the plot
                    st.pyplot(fig)

                    #Create a new column 'Date' by combining 'Year' and 'Month'
                    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))

                    #Group data by 'Date' and calculate the total number of visits
                    trend_data = df.groupby('Date').size().reset_index(name='Total Visits')

                    #Create a line plot
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(trend_data['Date'], trend_data['Total Visits'], color='red')

                    #Set title and labels
                    ax.set_title('Monthly Visits Trend from 2020 to 2024')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Total Visits')

                    #Format x-axis to display dates
                    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
                    plt.gcf().autofmt_xdate()

                    #Display the plot
                    st.pyplot(fig)

                    #Create a figure with 5 subplots (one for each year)
                    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 6))

                    #Loop through each year
                    for i, year in enumerate(range(2020, 2025)):
                        # Filter data for the current year
                        year_data = df[df['Year'] == year]

                        # Group data by 'Month' and calculate the total number of visits
                        month_data = year_data.groupby('Month').size().reset_index(name='Total Visits')

                        # Create a pie chart for the current year
                        axes[i].pie(month_data['Total Visits'], labels=month_data['Month'], autopct='%1.1f%%')
                        axes[i].set_title(f'{year}')

                    #Layout so plots do not overlap
                    fig.tight_layout()

                    #Display the plot
                    st.pyplot(fig)


                    #Group data by 'Year' and calculate the total number of visits
                    year_data = df.groupby('Year').size().reset_index(name='Total Visits')

                    #Create a pie chart
                    fig=plt.figure(figsize=(10, 8))
                    plt.pie(year_data['Total Visits'], labels=year_data['Year'], autopct='%1.1f%%')
                    plt.title('Yearly Visits')
                    plt.legend(title="Years", loc="upper right", bbox_to_anchor=(1.3, 1))

                    #Display the plot
                    st.pyplot(fig)

                    st.markdown('<hr style="border: 2px solid blue;">', unsafe_allow_html=True)

                    
                if st.markdown("<div style='text-align: center;'><button style='background-color: black; color: white; padding: 1px 400px; border: none; boarder-radius: 5px; cursor: pointer;' type='button' onclick='this.style.background=\"green\"'><u><b>TOTAL VISITS BY AGE RANGE FOR EACH YEAR</b></u></button></div>", unsafe_allow_html=True):
                    st.markdown("<h2 style='text-align: center;'></h2>", unsafe_allow_html=True)
                    #Step 2: Ensure date columns are in datetime format
                    df['VISIT DATE'] = pd.to_datetime(df['VISIT DATE'], errors='coerce')
                    df['DATE OF BIRTH'] = pd.to_datetime(df['DATE OF BIRTH'], errors='coerce')

                    #Step 3: Calculate Age at the Time of Visit
                    df['Age'] = df['VISIT DATE'].dt.year - df['DATE OF BIRTH'].dt.year

                    #Step 4: Categorize Age into Ranges
                    bins = [0, 18, 40, 59, float('inf')]  # Define age bins
                    labels = ['0-18', '19-40', '41-59', '60+']  # Corresponding age range labels
                    df['Age Range'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

                    #Step 5: Group by Year and Age Range
                    df['Year'] = df['VISIT DATE'].dt.year
                    visits_by_age_range = df.groupby(['Year', 'Age Range']).size().reset_index(name='Total Visits')

                    #Step 6: Pivot for Visualization
                    pivot_data = visits_by_age_range.pivot(index='Year', columns='Age Range', values='Total Visits').fillna(0)

                    #Step 7: Visualize Total Visits by Age Ranges for Each Year
                    fig, ax = plt.subplots(figsize=(14, 8))
                    pivot_data.plot(kind='bar', ax=ax, colormap='viridis')
                    ax.set_title('Total Visits by Age Range for Each Year', fontsize=16)
                    ax.set_xlabel('Year', fontsize=14)
                    ax.set_ylabel('Total Visits', fontsize=14)
                    ax.tick_params(axis='x',rotation=0)
                    ax.legend(title='Age Range', fontsize=12)
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.tight_layout()

                    #Display the plot
                    st.pyplot(fig)

                    #Step 1: Filter data by year and calculate total visits
                    years = [2020, 2021, 2022, 2023, 2024]
                    fig, axes = plt.subplots(nrows=1, ncols=len(years), figsize=(20, 6), sharey=True)

                    for i, year in enumerate(years):
                        year_data = df[df['Year'] == year]
                        age_counts = year_data['Age Range'].value_counts().reset_index()
                        age_counts.columns = ['Age Range', 'Total Visits']

                        ax = axes[i]
                        ax.pie(age_counts['Total Visits'], labels=age_counts['Age Range'], autopct='%1.1f%%', radius=1.2)
                        ax.set_title(f'{year}', fontsize=14)
                        

                    plt.tight_layout()
                    plt.suptitle('Age Distribution and Total Visits by Year', fontsize=16, y=1.02)
                    plt.subplots_adjust(top=0.85)

                    #Display the plot
                    st.pyplot(fig)

                    st.markdown('<hr style="border: 2px solid blue;">', unsafe_allow_html=True)


                if st.markdown("<div style='text-align: center;'><button style='background-color: black; color: white; padding: 1px 350px; border: none; boarder-radius: 5px; cursor: pointer;' type='button' onclick='this.style.background=\"green\"'><u><b>GENDER-SPECIFIC COUNTS OF REPEATED PATIENT VISITS</b></u></button></div>", unsafe_allow_html=True):
                    st.markdown("<h2 style='text-align: center;'></h2>", unsafe_allow_html=True)

                    #Standardize the SEX column
                    df['SEX'] = df['SEX'].str.strip().str.capitalize()
                    df['SEX'] = df['SEX'].replace({'F': 'Female', 'M': 'Male'})

                    #Convert VISIT DATE to datetime format
                    df['VISIT DATE'] = pd.to_datetime(df['VISIT DATE'], errors='coerce')

                    #Drop rows with invalid VISIT DATE values
                    df = df.dropna(subset=['VISIT DATE'])

                    #Filter data for years 2020–2024
                    df = df[(df['VISIT DATE'].dt.year >= 2020) & (df['VISIT DATE'].dt.year <= 2024)]

                    #Check if there are any rows after filtering
                    if df.empty:
                        st.warning("No data available for the years 2020–2024.")
                    else:
                        # Extract year and month from VISIT DATE
                        df['Year'] = df['VISIT DATE'].dt.year
                        df['Month'] = df['VISIT DATE'].dt.month

                        # Group data by Year, Month, and SEX, and count the number of visits
                        grouped = df.groupby(['Year', 'Month', 'SEX']).size().reset_index(name='Visit Count')

                        # Create a horizontal subplot for each year
                        years = sorted(df['Year'].unique())
                        fig, axes = plt.subplots(1, len(years), figsize=(20, 6), sharey=True)

                        # If there's only one year, `axes` will not be iterable, so we handle that case
                        if len(years) == 1:
                            axes = [axes]

                        for i, year in enumerate(years):
                            ax = axes[i]
                            year_data = grouped[grouped['Year'] == year]

                            # Pivot data for easier plotting
                            pivot = year_data.pivot(index='Month', columns='SEX', values='Visit Count').fillna(0)

                            # Plot data
                            pivot.plot(kind='bar', ax=ax, width=0.8)
                            ax.set_title(f'Visits by Sex - {year}')
                            ax.set_xlabel('Month')
                            ax.set_ylabel('Visit Count')
                            ax.set_xticks(range(12))
                            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
                            ax.legend(title='Sex')

                        # Adjust layout for better visualization
                        plt.tight_layout()

                        # Display the plot
                        st.pyplot(fig)

                    #Filter data for years 2020–2024
                    df_2020_2024 = df[(df['VISIT DATE'].dt.year >= 2020) & (df['VISIT DATE'].dt.year <= 2024)]

                    #Group data by Year and SEX, and count the number of visits
                    grouped_2020_2024 = df_2020_2024.groupby(['Year', 'SEX']).size().reset_index(name='Visit Count')

                    #Pivot data for easier plotting
                    pivot_2020_2024 = grouped_2020_2024.pivot(index='Year', columns='SEX', values='Visit Count').fillna(0)

                    #Create a figure and axis
                    fig, ax = plt.subplots(figsize=(12, 4))

                    #Plot data
                    pivot_2020_2024.plot(kind='bar', ax=ax)

                    #Set title and labels
                    ax.set_title('Male and Female Patients by Year (2020-2024)')
                    ax.set_xlabel('Year')
                    ax.set_ylabel('Visit Count')

                    #Display the plot
                    st.pyplot(fig)

                    #Filter data for years 2020–2024
                    df_2020_2024 = df[(df['VISIT DATE'].dt.year >= 2020) & (df['VISIT DATE'].dt.year <= 2024)]

                    #Group data by Year and SEX, and count the number of visits
                    grouped_2020_2024 = df_2020_2024.groupby(['Year', 'SEX']).size().reset_index(name='Visit Count')

                    #Filter data for female and male
                    female_data = grouped_2020_2024[grouped_2020_2024['SEX'] == 'Female']
                    male_data = grouped_2020_2024[grouped_2020_2024['SEX'] == 'Male']

                    #Create a figure and axis
                    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

                    #Plot female data
                    axes[0].pie(female_data['Visit Count'], labels=female_data['Year'], autopct='%1.1f%%')
                    axes[0].set_title('Female Total Visits by Year')

                    #Plot male data
                    axes[1].pie(male_data['Visit Count'], labels=male_data['Year'], autopct='%1.1f%%')
                    axes[1].set_title('Male Total Visits by Year')

                    #Display the plot
                    st.pyplot(fig)

                    st.markdown('<hr style="border: 2px solid blue;">', unsafe_allow_html=True)



                if st.markdown("<div style='text-align: center;'><button style='background-color: black; color: white; padding: 1px 350px; border: none; boarder-radius: 5px; cursor: pointer;' type='button' onclick='this.style.background=\"green\"'><u><b>NUMBER OF PATIENTS WITH REPEATED VISITS BY AGE GROUP</b></u></button></div>", unsafe_allow_html=True):
                    st.markdown("<h2 style='text-align: center;'></h2>", unsafe_allow_html=True)

                    #Convert 'DATE OF BIRTH' and 'VISIT DATE' to datetime
                    df['DATE OF BIRTH'] = pd.to_datetime(df['DATE OF BIRTH'])
                    df['VISIT DATE'] = pd.to_datetime(df['VISIT DATE'])

                    #Calculate age
                    current_date = datetime.now()
                    df['AGE'] = (current_date - df['DATE OF BIRTH']).dt.days // 365

                    #Define age groups
                    def categorize_age(age):
                        if age <= 18:
                            return '0-18'
                        elif age <= 40:
                            return '19-40'
                        elif age <= 59:
                            return '41-59'
                        else:
                            return '60+'

                    df['AGE GROUP'] = df['AGE'].apply(categorize_age)

                    #Count visits per patient
                    visit_counts = df.groupby(['FIRST NAME', 'SURNAME']).size().reset_index(name='VISIT COUNT')

                    #Merge visit counts back with age groups
                    visit_counts = visit_counts.merge(df[['FIRST NAME', 'SURNAME', 'AGE GROUP']], on=['FIRST NAME', 'SURNAME'])

                    #Group by age group and visit count
                    result = visit_counts.groupby(['AGE GROUP', 'VISIT COUNT']).size().unstack(fill_value=0)

                    #Reset index for plotting
                    result = result.reset_index()

                    #Plotting
                    fig, ax = plt.subplots(figsize=(12, 4))
                    sns.set(style="whitegrid")

                    #Melt the DataFrame for easier plotting
                    melted_result = result.melt(id_vars='AGE GROUP', var_name='VISIT COUNT', value_name='NUMBER OF PATIENTS')

                    #Create the bar plot
                    sns.barplot(data=melted_result, x='AGE GROUP', y='NUMBER OF PATIENTS', hue='VISIT COUNT', palette='viridis', ax=ax)

                    #Customize the plot
                    ax.set_title('Number of Patients with Repeated Visits by Age Group')
                    ax.set_xlabel('Age Group')
                    ax.set_ylabel('Number of Patients')
                    ax.legend(title='Visit Count', bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout()

                    #Display the plot
                    st.pyplot(fig)

                    st.markdown('<hr style="border: 2px solid blue;">', unsafe_allow_html=True)



                    
                if st.markdown("<div style='text-align: center;'><button style='background-color: black; color: white; padding: 1px 450px; border: none; boarder-radius: 5px; cursor: pointer;' type='button' onclick='this.style.background=\"green\"'><u><b>TOTAL VISITS BY GENDER</b></u></button></div>", unsafe_allow_html=True):
                    st.markdown("<h2 style='text-align: center;'></h2>", unsafe_allow_html=True)
                    #Group by FIRST NAME, SURNAME, and SEX to count visits
                    visit_counts = df.groupby(['FIRST NAME', 'SURNAME', 'SEX']).size().reset_index(name='VISIT COUNT')

                    #Create a new DataFrame to categorize visit counts by gender
                    gender_visit_counts = visit_counts.groupby(['SEX', 'VISIT COUNT']).size().reset_index(name='NUMBER OF PATIENTS')

                    #Pivot the DataFrame to have genders as columns
                    pivot_table = gender_visit_counts.pivot(index='VISIT COUNT', columns='SEX', values='NUMBER OF PATIENTS').fillna(0)

                    #Reset index to make 'VISIT COUNT' a column again
                    pivot_table = pivot_table.reset_index()

                    #Prepare data for plotting
                    visit_counts_unique = pivot_table['VISIT COUNT']
                    bar_width = 0.35  # Width of the bars
                    x = np.arange(len(visit_counts_unique))  # The label locations

                    #Plotting
                    fig, ax = plt.subplots(figsize=(12, 4))

                    #Plot bars for each gender
                    ax.bar(x - bar_width/2, pivot_table['Female'], width=bar_width, label='Female')
                    ax.bar(x + bar_width/2, pivot_table['Male'], width=bar_width, label='Male')

                    #Customize the plot
                    ax.set_title('Gender-Specific Counts of Repeated Patient Visits')
                    ax.set_xlabel('Number of Visits')
                    ax.set_ylabel('Number of Patients')
                    ax.set_xticks(x, visit_counts_unique)  # Set x-ticks to visit counts
                    ax.legend(title='Gender')
                    ax.grid(axis='y')

                    #Display the plot
                    st.pyplot(fig)

                    st.markdown('<hr style="border: 2px solid blue;">', unsafe_allow_html=True)

                if st.markdown("<div style='text-align: center;'><button style='background-color: black; color: white; padding: 1px 350px; border: none; boarder-radius: 5px; cursor: pointer;' type='button' onclick='this.style.background=\"green\"'><u><b>NUMBER OF PATIENTS BY REPEATED VISIT COUNTS (2020-2024)</b></u></button></div>", unsafe_allow_html=True):
                    st.markdown("<h2 style='text-align: center;'></h2>", unsafe_allow_html=True)
                    #Convert the 'VISIT DATE' column to datetime format
                    df['VISIT DATE'] = pd.to_datetime(df['VISIT DATE'])

                    #Create a new column for the year
                    df['YEAR'] = df['VISIT DATE'].dt.year

                    #Set up the figure and axes for subplots
                    fig, axs = plt.subplots(1, 5, figsize=(25, 5))

                    #Define the years for each subplot
                    years = [2020, 2021, 2022, 2023, 2024]

                    #Loop through each year and create a subplot
                    for i, year in enumerate(years):
                        # Filter data for the current year
                        yearly_data = df[df['YEAR'] == year]

                        # Group by FIRST NAME and SURNAME to count visits
                        visit_counts = yearly_data.groupby(['FIRST NAME', 'SURNAME']).size().reset_index(name='VISIT COUNT')

                        # Create a new DataFrame to categorize visit counts
                        visit_ranges = visit_counts['VISIT COUNT'].value_counts().reset_index()
                        visit_ranges.columns = ['VISIT COUNT', 'NUMBER OF PATIENTS']

                        # Sort the DataFrame by visit count
                        visit_ranges = visit_ranges.sort_values(by='VISIT COUNT')

                        # Plotting in the appropriate subplot
                        axs[i].bar(visit_ranges['VISIT COUNT'], visit_ranges['NUMBER OF PATIENTS'], color='purple')
                        axs[i].set_title(f'Year: {year}')
                        axs[i].set_xlabel('Number of Visits')
                        axs[i].set_ylabel('Number of Patients')
                        axs[i].grid(axis='y')

                    #Layout so plots do not overlap
                    fig.tight_layout()

                    #Display the plot
                    st.pyplot(fig)

                    st.markdown('<hr style="border: 2px solid blue;">', unsafe_allow_html=True)

                
    except pd.errors.EmptyDataError:
        st.warning("The uploaded file is empty. Please upload a valid CSV file.")
    except pd.errors.ParserError:
        st.warning("Error parsing the file. Please ensure it is a valid CSV.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning("Please upload a CSV file.")

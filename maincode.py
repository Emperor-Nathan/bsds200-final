import pandas as pd
import psycopg2 as psy
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import geopandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve
#from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer


r311C = reconfigureData(r311C) # this line and the next two are used to prep the data for the random forest
data = levelMaker(r311C, 'Borough') # this changes the string column into multiple columns of ints so the random forest can take the columns
data = fixNAs(data)  #gets rid of the NAs
forecast(data, 'agency_code', 'Taxi', 'CommonComplaint','0', '1', '2', '3', '4', '5', 'TimeRange') #Using a random forest and splitting the data into test and train, this function forecasts how long a complaint takes to be resolved. In the end it prints out the error of the prediction.


def forecast(df, x1, x2, x3, x4, x5, x6, x7, x8, x9, y):
    #ipdb.set_trace()
    df = df.loc[~(df.loc[:, [x1, x2, x3, y]].isna().any(axis=1)), :]
    x_train, x_test, y_train, y_test = train_test_split(df.loc[:, [x1, x2, x3, x4, x5, x6, x7, x8, x9]], (df.loc[:, [y]]), random_state=12)
    clf = RandomForestClassifier(random_state=120)
    clf.fit(x_train, y_train)
    clf_predict = clf.predict(x_test)
    error = str(round(MAPE(clf_predict, y_test.TimeRange), 2))
    print('The prediction model is incorrect '+error+'% of the time')

def levelMaker(data, col):
    enc = LabelBinarizer()
    enc.fit(data[col])
    transformed = enc.transform(data[col])
    ohe_df = pd.DataFrame(transformed)
    data = pd.concat([data, ohe_df], axis=1).drop([col], axis=1)
    return data

def fixNAs(data):
    data = data.loc[:,['agency_code', 'Taxi', 'CommonComplaint','0', '1', '2', '3', '4', '5', 'TimeRange']]
    data.loc[data.loc[:,'0'].isna(), '0'] = 0
    data.loc[data.loc[:,'1'].isna(), '1'] = 0
    data.loc[data.loc[:,'2'].isna(), '2'] = 0
    data.loc[data.loc[:,'3'].isna(), '3'] = 0
    data.loc[data.loc[:,'4'].isna(), '4'] = 0
    data.loc[data.loc[:,'5'].isna(), '5'] = 0
    return data


def reconfigureData(r311C):
    r311C = r311C.rename(columns = {'Created Date':'Create_dt', 'Closed Date':'Closed_dt',
                               'Taxi Company Borough':'TaxiComBoro','Taxi Pick Up Location':'TaxiPickLoc', 
                               'Complaint Type':'Complaint'})
    r311C['Create_dt'] = pd.to_datetime(r311C['Create_dt'], infer_datetime_format=True)
    r311C['Closed_dt'] = pd.to_datetime(r311C['Closed_dt'], infer_datetime_format=True)
    r311C['yr_diff'] = (r311C.Closed_dt).dt.year - (r311C.Create_dt).dt.year
    r311C = r311C.loc[(r311C.Closed_dt).dt.year>=2010 | (r311C.yr_diff>=0) ,:] # this is needed because there is a row that is a typo and says the closed date is in 1900 which is impossible since the created is 2013
    r311C['TimeRange'] = ((r311C.Closed_dt).dt.month - (r311C.Create_dt).dt.month) + (12*r311C.yr_diff)
    r311C = r311C.loc[(r311C.TimeRange>=0), :]
    r311C['TaxiComBoro'] = r311C['TaxiComBoro'].str.upper()
    r311C['Borough'] = r311C['Borough'].str.upper()
    r311C = r311C.loc[(r311C.Borough != 'NOT WITHIN NEW YORK CITY')| (r311C.TaxiComBoro != 'NOT WITHIN NEW YORK CITY')| (r311C.Borough != 'UNSPECIFIED')|(r311C.TaxiComBoro != 'UNSPECIFIED'),:]
    r311C.loc[r311C.TaxiComBoro=='BRONX', 'Borough'] = 'BRONX'
    r311C.loc[r311C.TaxiComBoro=='BROOKLYN', 'Borough'] = 'BROOKLYN'
    r311C.loc[r311C.TaxiComBoro=='MANHATTAN', 'Borough'] = 'MANHATTAN'
    r311C.loc[r311C.TaxiComBoro=='STATEN ISLAND', 'Borough'] = 'STATEN ISLAND'
    r311C.loc[r311C.TaxiComBoro=='QUEENS', 'Borough'] = 'QUEENS'
    r311C.loc[r311C.TaxiComBoro.isna(), 'Taxi'] = 0
    r311C.loc[~(r311C.TaxiComBoro.isna()), 'Taxi'] = 1
    r311C.loc[r311C.Agency == '3-1-1', 'agency_code'] = 1
    r311C.loc[r311C.Agency == 'DCA', 'agency_code'] = 2
    r311C.loc[r311C.Agency == 'DEP', 'agency_code'] = 3
    r311C.loc[r311C.Agency == 'DFTA', 'agency_code'] = 4
    r311C.loc[r311C.Agency == 'DHS', 'agency_code'] = 5
    r311C.loc[r311C.Agency == 'DOB', 'agency_code'] = 6
    r311C.loc[r311C.Agency == 'DOE', 'agency_code'] = 7
    r311C.loc[r311C.Agency == 'DOF', 'agency_code'] = 8
    r311C.loc[r311C.Agency == 'DOHMH', 'agency_code'] = 9
    r311C.loc[r311C.Agency == 'DOITT', 'agency_code'] = 10
    r311C.loc[r311C.Agency == 'DOT', 'agency_code'] = 11
    r311C.loc[r311C.Agency == 'DPR', 'agency_code'] = 12
    r311C.loc[r311C.Agency == 'DSNY', 'agency_code'] = 13
    r311C.loc[r311C.Agency == 'HPD', 'agency_code'] = 14
    r311C.loc[r311C.Agency == 'HRA', 'agency_code'] = 15
    r311C.loc[r311C.Agency == 'NYCEM', 'agency_code'] = 16
    r311C.loc[r311C.Agency == 'NYPD', 'agency_code'] = 17
    r311C.loc[r311C.Agency == 'TLC', 'agency_code'] = 18
    r311C.loc[:, 'CommonComplaint'] = 0
    r311C.loc[(r311C.Complaint.str.find('Noise')>=0) | (r311C.Complaint.str.find('Parking')>=0) | (r311C.Complaint.str.find('Blocked Driveway')>=0), 'CommonComplaint'] = 1
    return r311C


def MAPE(predictions, targets):
    len = predictions.shape[0]
    targets = np.array(targets)
    sm = 0
    for i in range (len):
        if (predictions[i]==0):
            x = (np.absolute(predictions[i]-targets[i]))
        else:
            x = (np.absolute(predictions[i]-targets[i]))/predictions[i]        
        sm = sm + x
    return ((sm*100)/len)

#sprint 3 code:

def taxi_plot(base_dir, CONNECTION_STRING):
    r311_1 = loadSQL(CONNECTION_STRING)
    r311C = r311_1.copy()
    r311C = reconfigureData(r311C)
    r311noTaxis = r311C.loc[(r311C.TaxiComBoro.isna())] #separates the data away from taxis
    r311Taxis = r311C.loc[~(r311C.TaxiComBoro.isna())] # separates the data to only include taxis
    r311noTaxis = r311noTaxis.loc[:,['Created_dt', 'Closed_dt', 'Borough', 'Latitude', 'Longitude','TimeRange']]
    r311Taxis = r311Taxis.loc[:,['Created_dt', 'Closed_dt', 'TaxiComBoro', 'TaxiPickLoc', 'Latitude', 'Longitude','TimeRange']]
    NTBorotime = r311noTaxis.groupby('Borough', as_index=False).agg({'TimeRange':'mean'}) #gets the mean for all non taxi data by borough
    TBorotime = r311Taxis.groupby('TaxiComBoro', as_index=False).agg({'TimeRange':'mean'}) #gets the mean for all taxi data by borough
    NTBorotime = NTBorotime.loc[(NTBorotime.Borough != 'NOT WITHIN NEW YORK CITY')&(NTBorotime.Borough != 'UNSPECIFIED'),:].reset_index(drop='true') #gets rid of useless data
    TBorotime = TBorotime.loc[(TBorotime.TaxiComBoro != 'NOT WITHIN NEW YORK CITY'),:].reset_index(drop='true')
    nyc = geopandas.read_file(base_dir + '/Borough.geojson') #this loads a map of nyc and is broken down into boroughs
    #the next few lines gather data based on borough
    nyc.loc[(TBorotime.TaxiComBoro == 'BRONX'), 'diff'] = (TBorotime.loc[(TBorotime.TaxiComBoro == 'BRONX'),'TimeRange'] - NTBorotime.loc[(NTBorotime.Borough == 'BRONX'),'TimeRange'])
    nyc.loc[(TBorotime.TaxiComBoro == 'BROOKLYN'),'diff'] = TBorotime.loc[(TBorotime.TaxiComBoro == 'BROOKLYN'),'TimeRange'] - NTBorotime.loc[(NTBorotime.Borough == 'BROOKLYN'),'TimeRange']
    nyc.loc[(TBorotime.TaxiComBoro == 'MANHATTAN'),'diff'] = TBorotime.loc[(TBorotime.TaxiComBoro == 'MANHATTAN'),'TimeRange'] - NTBorotime.loc[(NTBorotime.Borough == 'MANHATTAN'),'TimeRange']
    nyc.loc[(TBorotime.TaxiComBoro == 'QUEENS'), 'diff'] = TBorotime.loc[(TBorotime.TaxiComBoro == 'QUEENS'),'TimeRange'] - NTBorotime.loc[(NTBorotime.Borough == 'QUEENS'),'TimeRange']
    nyc.loc[(TBorotime.TaxiComBoro == 'STATEN ISLAND'), 'diff'] = TBorotime.loc[(TBorotime.TaxiComBoro == 'STATEN ISLAND'),'TimeRange'] - NTBorotime.loc[(NTBorotime.Borough == 'STATEN ISLAND'),'TimeRange']
    plt.figure(figsize=(435,290)) # makes sure the data fits
    nyc.plot(column='diff', cmap='Greens', legend = 'true', edgecolor='black')
    plt.title("Taxis's effects on the length of a request")
    plt.axis('off') #removes the axis tick marks and makes it look cleaner
    plt.savefig(base_dir+'taxis.png') #saves the plot in the directory that the data is in
    plt.show()
###

def getRandom():
    if random.randint(0, 8e6) < 1e5:
        return True
    return False

def ComplaintTypePlot(base_dir):
    r311_2 = pd.read_csv(base_dir + '/311_Service_Requests.csv', nrows=100000)
    ans1 = r311_2.loc[:, ['Complaint Type', 'Status']].groupby('Complaint Type').agg('count')[['Status']].sort_values('Status', ascending = False)
    t1 = r311_2.loc[(r311_2['Status'] == 'Closed'), ['Complaint Type', 'Unique Key']].groupby('Complaint Type').agg('count')[['Unique Key']].rename(columns={'Unique Key':'closed'})
    t2 = r311_2.loc[(r311_2['Status'] != 'Closed'), ['Complaint Type', 'Unique Key']].groupby('Complaint Type').agg('count')[['Unique Key']].rename(columns={'Unique Key':'notclosed'})
    ans2 = pd.concat([t1, t2], axis = 1)
    plt.figure(figsize=(435,290))
    plt.title('Reported incidents by type')
    ans1C = ans1.loc[(ans1['Status'] > 3000), ['Status']]
    plt.ylabel('NUmber of requests / 100')
    plt.bar(ans1C.index, ans1C['Status']/100, 0.2)
    plt.show()

###


def agency_plot(base_dir):
    #read and then copy file
    r311_1 = pd.read_csv(base_dir + '/311_Service_Requests.csv', nrows=10000)
    r311c = r311_1.copy()
    #rename certain columns so there is no white-space in names
    r311c = r311c.rename(columns = {'Agency' : 'Agency_Name', 'Complaint Type' : 'Complaint_Type'})
    #narrowing data to a few columns
    r311c = r311c.loc[:,['Agency_Name', 'Complaint_Type', 'Status']]
    #adding column for when case is closed and not closed
    r311c.loc[(r311c.Status == 'Closed', 'Closed_Flag')] = 1
    r311c.loc[(r311c.Status != 'Closed', 'Closed_Flag')] = 0
    #groupby agency name to see completion rate
    r311c = r311c.loc[:, ['Agency_Name', 'Closed_Flag']].groupby('Agency_Name', as_index = False).agg({'Closed_Flag' : 'mean'}).sort_values('Closed_Flag', ascending = True)
    #gets all that don't have 100% completion
    ans = r311c.loc[(r311c.Closed_Flag < 1), ['Agency_Name', 'Closed_Flag']]
    #desides plot size and and customizations
    plt.figure(figsize=(6,3))
    plt.title('Ratio of Completions for Agencies That Do Not Have 100% Completion')
    #import ipdb
    #ipdb.set_trace()
    plt.ylabel('Ratio of Closed Requests')
    plt.bar(ans.Agency_Name, ans.Closed_Flag, .2)
    #shows plot
    plt.show()



#code for sprint 2:


def load_pandas(base_dir):    
    R311_FILE_LOCATIONS = base_dir               
    FINAL_FILE = '311_Service_Requests.csv' #the file is saved on my computer as a csv file
    DF_to_Return = pd.read_csv( R311_FILE_LOCATIONS+FINAL_FILE, nrows=8000000, sep=',')    #this is only the first 1/3ish of the data but it is the most I can load into one

    r311_2 = pd.read_csv( R311_FILE_LOCATIONS+FINAL_FILE, skiprows=8000000, nrows=8000000) # this is the middle 1/3ish

    r311_3 = pd.read_csv( R311_FILE_LOCATIONS+FINAL_FILE, skiprows=16000000) #this is the last 1/3ish

    return DF_to_Return # we are only returning 1/3 of the data because if I merge any of 2 of the 3 subsets together my computer crashes due to a memory error


def loadSQL(CONNECTION_STRING):
    SQLConn = psy.connect(CONNECTION_STRING)
    SQLCursor = SQLConn.cursor()    
    schema_name = 'nyc_311'    
    table_name = 'requests'    
    try:        
        SQLCursor.execute("""DROP TABLE %s.%s;""" % (schema_name, table_name))        
        SQLConn.commit()    
    except psy.ProgrammingError:        
        print("CAUTION: Tablenames not found: %s.%s" % (schema_name, table_name))        
        SQLConn.rollback()    
    SQLCursor = SQLConn.cursor()    
    SQLCursor.execute("""            
            CREATE TABLE %s.%s            
            (Index int
            , Unique_Key varchar(8)
            , Created_Date date
            , Closed_Date date
            , Agency varchar(15)
            , Agency_Name varchar(90)
            , Complaint_Type varchar(60)
            , Descriptor varchar(128)
            , Location_Type varchar(73)
            , Incident_Zip varchar(18)
            , Incident_Address varchar(109)
            , Street_Name varchar(110)
            , Cross_Street_1 varchar(110)
            , Cross_Street_2 varchar(110)
            , Intersection_Street_1 varchar(110)
            , Intersection_Street_2 varchar(110)
            , Address_Type varchar(24)
            , City varchar(31)
            , Landmark varchar(48)
            , Facility_Type varchar(20)
            , Status varchar(20)
            , Due_Date date
            , Resolution_Description varchar(751)
            , Resolution_Action_Updated_Date date
            , Community_Board varchar(35)
            , BBL float
            , Borough varchar(25)
            , X_Coordinate_State_Plane float
            , Y_Coordinate_State_Plane float
            , Open_Data_Channel_Type varchar(9)
            , Park_Facility_Name varchar(98)
            , Park_Borough varchar(30)
            , Vehicle_Type varchar(37)
            , Taxi_Company_Borough varchar(25)
            , Taxi_Pick_Up_Location varchar(92)
            , Bridge_Highway_Name varchar(88)
            , Bridge_Highway_Direction varchar(62)
            , Road_Ramp varchar(10)
            , Bridge_Highway_Segment varchar(119)
            , Latitude float
            , Longitude float
            , Location varchar(47)
            );""" % (schema_name, table_name))
    SQL_STATEMENT = f"""        
        COPY {schema_name}.{table_name} FROM STDIN WITH            
        CSV             
        HEADER            
        DELIMITER AS E',';       
    """   
    SQLConn.commit()     
    return pandas.io.sql.read_sql_query(SQL_STATEMENT, SQLConn)


def test(base_dir, CONNECTION_STRING):
    #first test
    TEMPORARY_DIR = base_dir + '/TmpDir/'
    DF = pd.read_csv( TEMPORARY_DIR + 'Final_file.tdf', sep='\t')
    SQLConn = psycopg2.connect(CONNECTION_STRING)
    SQLCursor = SQLConn.cursor()

    #SQL QUERY
    SQLCursor.execute("""Select count(*) from project1.NYC_311;""")
    #EXECUTES SQL QUERY
    sql_rows =SQLCursor.fetchall()
    #GETS THE DATA FROM THE DF
    sql_rows = sql_rows[0][0]
    #PANDAS DF COUNT
    DF_rows = DF.shape[0]
    #CHECKING IF THE PANDAS DF IS THE SAME SIZE AS THE SQL DF
    return (DF_rows == sql_rows)

def test2(dir, connection_string):
    pandasDF = load_pandas(dir)
    numRowsP = pandasDF.agg('count')[['Unique Key']]
    sqlDF = load_SQL(dir)
    SQLConn = psy.connect(connection_string)
    SQLCursor = SQLConn.cursor()
    numRowsS = SQLCursor.execute("""select count(Unique_Key) from Project5.nyc_311;""")
    SQLConn.commit()
    return (numRowsP == numRowsS)

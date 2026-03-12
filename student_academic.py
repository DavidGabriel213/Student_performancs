import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("/storage/emulated/0/Download/student_academic_messy.csv")
df=df.drop_duplicates()
df["StudentID"]=df["StudentID"].astype(str).str.strip()
df["Age"]=pd.to_numeric(df["Age"],errors="coerce")
q1=df["Age"].quantile(0.25)
q3=df["Age"].quantile(0.75)
iqr=q3-q1
df["Age"]=df["Age"].apply(lambda x: np.nan if (x>(q3+1.5*iqr) or x<(q3-1.5*iqr)) else x)
df["Age"]=np.abs(df["Age"].fillna(df["Age"].median())).astype(int)
df["Gender"]=df["Gender"].astype(str).str.strip()
def gender_correction(c):
    d={"F":"Female","M":"Male","nan":np.nan,"male":"Male","FEMALE":"Female"}
    key=list(d.keys())
    if c in key:
        return d[c]
    else:
        return c
df["Gender"]=df["Gender"].apply(lambda x: gender_correction(x))
df["Gender"]=df["Gender"].fillna(df['Gender'].mode()[0])
df["State"]=df["State"].astype(str).str.strip()
df["State"]=df["State"].fillna(df["State"].mode()[0])
def university_correction(c):
    d={"oau":"OAU Ile-Ife","atbu":"ATBU Bauchi","buk":"BUK Kano","unilag":"University of Lagos","UI":"University of Ibadan","futa":"FUTA","ABU":"ABU Zaria","nan":np.nan}
    key=list(d.keys())
    if c in key:
        return d[c]
    else:
        return c
df["University"]=df["University"].astype(str).str.strip()
df["University"]=df["University"].apply(lambda x: university_correction(x))
df["University"]=df["University"].fillna(df.groupby("State")["University"].transform(lambda x: x.mode()[0]))
def course_correction(c):
    d={"LAW":"Law","CS":"Computer Science","ARCH":"Architecture","STAT":"Statistics","PHY":"Physics","MED":"Medicine","ACCT":"Accounting","nan":np.nan}
    key=list(d.keys())
    if c in key:
        return d[c]
    else:
        return c
df["Course"]=df["Course"].astype(str).str.strip()
df["Course"]=df["Course"].apply(lambda x: course_correction(x))
df["Course"]=df["Course"].fillna(df.groupby("University")["Course"].transform(lambda x:x.mode()[0]))
df["YearOfStudy"]=df["YearOfStudy"].astype(str).str.replace("L","").str.strip()
df["YearOfStudy"]=df["YearOfStudy"].apply(lambda x: np.nan if x=="nan" else x[0] if len(x)>1 else x)
df["YearOfStudy"]=np.abs(pd.to_numeric(df["YearOfStudy"],errors="coerce"))
df["YearOfStudy"]=df["YearOfStudy"].apply(lambda x: np.nan if x>5 else x)
df["YearOfStudy"]=df["YearOfStudy"].fillna(df["YearOfStudy"].median())
df["YearOfStudy"]=df["YearOfStudy"].astype(int)
df["CGPA"]=df["CGPA"].astype(str).str.replace("/5.0","").str.replace("-","").str.strip()
df["CGPA"]=pd.to_numeric(df["CGPA"],errors="coerce")
df["CGPA"]=df["CGPA"].apply(lambda x: np.nan if x>5 else x)
k=df.groupby("Course")["CGPA"].mean()
df=df.set_index("Course")
df["CGPA"]=df["CGPA"].fillna(k)
df=df.reset_index()
df["CGPA"]=df["CGPA"].round(2)
df["AttendanceRate(%)"]=df["AttendanceRate(%)"].astype(str).str.replace('-',"").str.replace("%","").str.strip()
df["AttendanceRate(%)"]=pd.to_numeric(df["AttendanceRate(%)"],errors="coerce")
df["AttendanceRate(%)"]=df["AttendanceRate(%)"].apply(lambda x:x/10 if x>100 else x)
df["AttendanceRate(%)"]=(df["AttendanceRate(%)"].fillna(df["AttendanceRate(%)"].mean())).round(1)
df["DailyStudyHours"]=df["DailyStudyHours"].astype(str).str.replace("hrs","").str.replace("-","").str.strip()
df["DailyStudyHours"]=pd.to_numeric(df["DailyStudyHours"],errors="coerce")
Q1=df["DailyStudyHours"].quantile(0.25)
Q3=df["DailyStudyHours"].quantile(0.75)
IQR=(Q3-Q1)
df["DailyStudyHours"]=df["DailyStudyHours"].apply(lambda x: np.nan if (x>(Q3+1.5*IQR) or x<(Q3-1.5*IQR)) else x)
df["DailyStudyHours"]=(df["DailyStudyHours"].fillna(df["DailyStudyHours"].mean())).round(1)
df["AssignmentScore(%)"]=df["AssignmentScore(%)"].astype(str).str.replace('-',"").str.replace("%","").str.strip()
df["AssignmentScore(%)"]=pd.to_numeric(df["AssignmentScore(%)"],errors="coerce")
df["AssignmentScore(%)"]=df["AssignmentScore(%)"].apply(lambda x:x/10 if x>100 else x)
df["AssignmentScore(%)"]=(df["AssignmentScore(%)"].fillna(df["AssignmentScore(%)"].mean())).round(1)
df["ExamScore(%)"]=df["ExamScore(%)"].astype(str).str.replace('-',"").str.replace("%","").str.strip()
df["ExamScore(%)"]=pd.to_numeric(df["ExamScore(%)"],errors="coerce")
df["ExamScore(%)"]=df["ExamScore(%)"].apply(lambda x:x/10 if x>100 else x)
df["ExamScore(%)"]=(df["ExamScore(%)"].fillna(df["ExamScore(%)"].mean())).round(1)
df["DistanceFromCampus(km)"]=df["DistanceFromCampus(km)"].astype(str).str.replace("-","").str.replace("km","").str.strip()
df["DistanceFromCampus(km)"]=pd.to_numeric(df["DistanceFromCampus(km)"],errors="coerce")
m=df.groupby("University")["DistanceFromCampus(km)"].mean()
df=df.set_index("University")
df["DistanceFromCampus(km)"]=(df["DistanceFromCampus(km)"].fillna(m)).round(1)
df=df.reset_index()
def scholarship_correction(c):
    d={"yes":"Yes","no":"No","1":"Yes","0":"No","nan":np.nan}
    key=list(d.keys())
    if c in key:
        return d[c]
    else:
        return c
df["HasScholarship"]=df["HasScholarship"].astype(str).str.strip()
df["HasScholarship"]=df["HasScholarship"].apply(lambda x: scholarship_correction(x))
df["HasScholarship"]=df["HasScholarship"].fillna(df.groupby("University")["HasScholarship"].transform(lambda x:x.mode()[0]))
def ptjob_correction(c):
    d={"yes":"Yes","no":"No","1":"Yes","0":"No","nan":np.nan}
    key=list(d.keys())
    if c in key:
        return d[c]
    else:
        return c
df["PartsTimeJob"]=df["PartsTimeJob"].astype(str).str.strip()
df["PartsTimeJob"]=df["PartsTimeJob"].apply(lambda x: ptjob_correction(x))
df["PartsTimeJob"]=df["PartsTimeJob"].fillna(df.groupby("State")["PartsTimeJob"].transform(lambda x:x.mode()[0]))
def ITaccess_correction(c):
    d={"yes":"Yes","no":"No","1":"Yes","0":"No","nan":np.nan}
    key=list(d.keys())
    if c in key:
        return d[c]
    else:
        return c
df["InternetAccess"]=df["InternetAccess"].astype(str).str.strip()
df["InternetAccess"]=df["InternetAccess"].apply(lambda x: ITaccess_correction(x))
df["InternetAccess"]=df["InternetAccess"].fillna(df.groupby("University")["InternetAccess"].transform(lambda x:x.mode()[0]))
def pass_correction(c):
    d={"yes":"Yes","no":"No","1":"Yes","0":"No","nan":np.nan}
    key=list(d.keys())
    if c in key:
        return d[c]
    else:
        return c
df["Passed"]=df["Passed"].astype(str).str.strip()
df["Passed"]=df["Passed"].apply(lambda x: pass_correction(x))
df.loc[df["Passed"].isnull(), "Passed"] = np.where(df.loc[df["Passed"].isnull(),"CGPA"]>=1.5,"Yes","No")
df["PerformanceIndex"]=((df["ExamScore(%)"]*0.5) + (df["AssignmentScore(%)"]*0.3) + (df["AttendanceRate(%)"]*0.2)).round(2)
df["StudyEfficiency"]=(df["CGPA"]/df["DailyStudyHours"]).round(2)
df["Class"]=df["CGPA"].apply(lambda x: "First Class" if x>=4.5 else "Second Class Upper" if x>=4 else "Second Class Lower" if x>=3.5 else "Third Class" if x>=2.5 else "Pass")
fig,ax=plt.subplots(2,2,figsize=(9,9))
p_index=df.groupby("Course")["PerformanceIndex"].mean()
p_index.plot(kind="bar",ax=ax[0,0],color="purple")
ax[0,0].set_ylabel("AveragePerformanceIndex")
ax[0,0].set_title("AveragePerformanceIndexby Course",color="red")
p_efficiency=df.groupby("University")["StudyEfficiency"].mean()
p_efficiency.plot(kind="bar",ax=ax[0,1])
ax[0,1].set_ylabel("Av.StudyEfficiency")
ax[0,1].set_title("Av.StudyEffi. by Uni",color="red")
bine=np.linspace(0,5,20, endpoint=True)
ax[1,0].hist(df["CGPA"],bins=bine,color="green")
ax[1,0].set_ylabel('Frequency')
ax[1,0].set_xlabel("CGPA")
ax[1,0].set_title("CGPA distribution",color="red")
df["PassedCode"]=df["Passed"].apply(lambda x: 1 if x=="Yes" else 0)
ax[1,1].scatter(df["AttendanceRate(%)"],df["ExamScore(%)"],c=df["PassedCode"],cmap="Set3")
ax[1,1].set_ylabel("ExamsScore")
ax[1,1].set_xlabel("AttendanceRate(%)")
ax[1,1].set_title("Scatter plot",color="red")
from matplotlib.patches import Patch
legend = [Patch(color='yellow', label='Fail'), Patch(color='lightgreen', label='Pass')]
ax[1,1].legend(handles=legend)
h=df.groupby("InternetAccess")["CGPA"].mean().round(2)
print("Average CGPA base on Internet Access",h)
if h.loc["Yes"]>h.loc["No"]:
    print("Yes having Internet connection Improves CGPA")
else:
    print("Having Internet Access doesnt improve CGPA")
b=df.groupby("HasScholarship")["DailyStudyHours"].mean().round(1)
print("Average Study Hours base on Schorlaship",b)
if b.loc["Yes"]>b.loc["No"]:
    print("Yes Scholarship Student study more hours on average")
else:
    print("No Scholarship Student don't study more hours")
cova=np.cov(df["DistanceFromCampus(km)"],df["AttendanceRate(%)"])
corr=(cova/(df["DistanceFromCampus(km)"].std()*df["AttendanceRate(%)"].std())).round(3)
print("The Correlation Between Distance From campus and Attendance rate is ",corr[0,1])
plt.tight_layout()
plt.show()

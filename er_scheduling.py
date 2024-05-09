#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[1]:


from ortools.sat.python import cp_model
from datetime import datetime
from datetime import timedelta  
import pandas as pd
import math
import csv
import os
import time

from datetime import date
import holidays

import numpy as np
from tabulate import tabulate

import matplotlib.pyplot as plt


# ### Global Constants

# In[2]:


DATE_FORMAT = "%Y-%m-%d"
SIX_MONTHS = 6 * 30
today = datetime.today()


# In[3]:


canada_ontario_holidays = holidays.CA(prov="ON")
print(date(2020, 9, 7) in holidays.CA())


# In[4]:


holidays_ontario_canada_dic = holidays.CA(prov="ON", years=[year for year in range(2020, 2041)])
holidays_ontario_canada_items = holidays_ontario_canada_dic.items()
holidays_ontario_canada_date = list(holidays_ontario_canada_dic.keys())


# In[5]:


holidays_ontario_canada_date


# In[6]:


for d in holidays_ontario_canada_date:
    print(d)


# In[7]:


for h in holidays_ontario_canada_items:
    print(h)


# ### Get start and End Date for the scheduler

# In[8]:


scheduler_start_date = "2020-12-23"
scheduler_start_date = datetime.fromisoformat(scheduler_start_date)


# ### Read CSV Files

# The CSV file should be clean. Please remove the deleted rows and columns clearly.

# In[9]:


doctors = pd.read_csv("doctors.csv")
doctors["firstname"]


# In[10]:


shifts = pd.read_csv("shifts.csv")
shifts["name"]


# ### Output File

# In[11]:


# output_file = open("schedules.txt", "w")


# ## Initialize Variables

# <font color='red'>Wraning: The number of days for the scheduler needs to be at least 7 days</font>

# In[12]:


num_doctors = len(doctors)
num_days = 30
num_shifts = len(shifts)


# In[13]:


model = cp_model.CpModel()


# ### Initialize OR-Tools Variables

# In[14]:


doc_shifts = {}
for n in range(num_doctors):
    for d in range(num_days):
        for s in range(num_shifts):
            doc_shifts[(n,d,s)] = model.NewBoolVar("shift_n%i_d%i_s%i" % (n,d,s))


# ## Functions

# ### Get the day of week
# get the weekday from day
# 
# <b>Arguments:</b>
# 
# day (the number of days after scheduler start date) 
# 
# <b>Returns:</b>
# 
# day of week (Monday is 0 and Sunday is 6)

# In[15]:


def day_to_weekday(day):
    day_date = scheduler_start_date + timedelta(days=day)
    return day_date.weekday()


# ### Is Weekend
# is day weekend (Saturday or Sunday)?
# 
# <b>Arguments:</b>
# 
# day (starts from the scheduler date. e.g. 1) 
# 
# <b>Returns:</b>
# 
# 1 if weekend, 0 if is not weekend 

# In[16]:


def is_weekend(day):
    weekday = day_to_weekday(day)
    return 1 if (weekday == 5 or weekday == 6) else 0


# ### Is Weekday
# is day weekday (Monday, Tuesday, Wednesday, Thursday, Friday)?
# 
# <b>Arguments:</b>
# 
# day (starts from the scheduler date. e.g. 1) 
# 
# <b>Returns:</b>
# 
# 1 if weekday, 0 if is not 

# In[17]:


def is_weekday(day):
    weekday = day_to_weekday(day)
    return 1 if (weekday == 0 or weekday == 1 or weekday == 2 or weekday == 3 or weekday == 4) else 0


# ### Is Saturday
# is day weekend Saturday?
# 
# <b>Arguments:</b>
# 
# day (starts from the scheduler date. e.g. 1) 
# 
# <b>Returns:</b>
# 
# 1 if saturday, 0 if it is not 

# In[18]:


def is_saturday(day):
    weekday = day_to_weekday(day)
    return 1 if (weekday == 5) else 0


# ### Date To Day
# convert date to day
# 
# <b>Arguments:</b>
# 
# date (format: . e.g. 1) 
# 
# <b>Returns:</b>
# 
# the number of days after scheduler start date

# In[19]:


def date_to_day(date):
    diff = date - scheduler_start_date.date()
    return diff.days


# In[20]:


def datetime_to_day(datetime):
    diff = datetime.date() - scheduler_start_date.date()
    return diff.days


# In[21]:


print(datetime_to_day(today))


# In[22]:


print(date_to_day(holidays_ontario_canada_date[0]))


# ### get_doc_lastname

# In[23]:


def get_doc_lname(n):
    return doctors["lastname"][n]


# ### get_date

# In[24]:


def get_date(d):
    return scheduler_start_date.date() + timedelta(days=d)


# ### get_shift_name

# In[25]:


def get_shift_name(s):
    return shifts["name"][s]


# ### print_shift_status

# In[26]:


def get_shift_status(solver, doc_shifts, d, s):
    for n in range(num_doctors):
        if solver.Value(doc_shifts[(n, d, s)]) == 1:
            print("%s: Shift %s -> Doctor %s" % (get_date(d), get_shift_name(s), get_doc_lname(n)))


# ### get_sum_working_days

# returns the number of days the given doctor "n" is scheduled to work

# In[27]:


def get_sum_working_days(solver, doc_shifts, n):
    return sum([solver.Value(doc_shifts[(n,d,s)]) for d in range(num_days) for s in range(num_shifts)])


# ### get_sum_working_shift

# returns the number of days the given doctor "n" is scheduled to work on the given shift "s"

# In[28]:


def get_sum_working_shift(solver, doc_shifts, n, s):
    return sum([solver.Value(doc_shifts[(n,d,s)]) for d in range(num_days)])


# ### get_sum_working_holidays

# returns the number of days the given doctor "n" is scheduled to work on holidays

# In[29]:


def get_sum_working_holidays(solver, doc_shifts, holiday_days, n):
    return sum([solver.Value(doc_shifts[(n,d,s)]) for d in holiday_days for s in range(num_shifts)])


# ### get_day_shifts

# returns the index of days shifts (list)

# In[30]:


def get_day_shifts():
    day_shifts = []
    for s in range(num_shifts):
        start_time = shifts['start-time'][s]
        if start_time >= 700 and start_time <= 1200:
            day_shifts.append(s)
    return day_shifts


# In[31]:


# get_day_shifts()


# ### get_sum_day_shifts

# returns the number of days the given doctor "n" is scheduled to work on day shifts

# In[32]:


def get_sum_day_shifts(solver, doc_shifts, n):
    day_shifts = get_day_shifts()
    return sum([solver.Value(doc_shifts[(n,d,s)]) for d in range(num_days) for s in get_day_shifts()])


# ## Solution Visualization

# ### Limited Solution Printer

# In[33]:


class DoctorsPartialSolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, doc_shifts, limit):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__doc_shifts = doc_shifts
        self.__solution_count = 0
        self.__solution_limit = limit

    def on_solution_callback(self):
        self.__solution_count += 1
        print("Solution %d" % self.__solution_count)
        day_list = []
        for d in range(num_days):
            shift_list = []
#             shift_list.append("Day%d" % d)
            shift_list.append((scheduler_start_date + timedelta(days=d)).strftime("%d %b, %Y"))
            for s in range(num_shifts):
                doc = -1
                for n in range(num_doctors):
                    if self.Value(self._doc_shifts[(n,d,s)]):
                        doc = n
#                 shift_list.append("Doctor%d" % doc)
                shift_list.append(doctors["firstname"][doc] + " " + doctors["lastname"][doc])
            day_list.append(shift_list)
            
#         headers = [("Shift%d" % s) for s in range(self._num_shifts)]
        headers = ["Shift %d" % shifts["name"][s] for s in range(self._num_shifts)]
        table = tabulate(day_list, headers, tablefmt="fancy_grid")
        print(table)
        
        print("------------------------------------------------------------------")
        if self.__solution_count >= self.__solution_limit:
            self.StopSearch()

    def solution_count(self):
        return self.__solution_count


# ## Hard Rules

# ### Hard Rule 1

# Maximum work 7 consecutive days

# In[34]:


for n in range(num_doctors):
    for d in range(num_days - 8 + 1):
        model.Add(sum(doc_shifts[(n,d+d_f,s)] for s in range(num_shifts) for d_f in range(8)) <= 7)


# ### Hard Rule 2

# No two shifts in a single day

# In[35]:


for n in range(num_doctors):
    for d in range(num_days):
        model.Add(sum(doc_shifts[(n,d,s)] for s in range(num_shifts)) <= 1)


# ### Hard Rule 3

# 2 days off after last midnight (except on call shift)

# In[36]:


is_working = {}
for n in range(num_doctors):
    for d in range(num_days):
        is_working[(n,d)] = model.NewBoolVar("doc%i_working_day%i" % (n, d))
        model.Add(is_working[(n,d)] == sum(doc_shifts[(n,d,s)] for s in range(num_shifts)))


# In[37]:


is_working_midnight = {}

midnight_shifts = shifts["midnight"]
for n in range(num_doctors):
    for d in range(num_days):
        for s in range(num_shifts):
            if midnight_shifts[s] == 1:
                is_working_midnight[(n, d)] = model.NewBoolVar("is_working_midnight_doc%i-day%i" % (n, d))
                model.Add(is_working_midnight[(n, d)] == doc_shifts[(n, d, s)])


# In[38]:


off_two_days = {}

for n in range(num_doctors):
    for d in range(num_days-1):
        off_two_days[(n, d)] = model.NewBoolVar("off_two_days_doc%i-day%i" % (n, d))
        model.Add((is_working[(n,d)] + is_working[(n,d+1)]) == 0).OnlyEnforceIf(off_two_days[(n, d)])


# In[39]:


on_call_shifts = shifts["on-call"]  
midnight_shifts = shifts["midnight"]

# if night shift exists
if 1 in midnight_shifts.values:
    for n in range(num_doctors):
        for d in range(num_days-2):

            model.AddBoolOr([is_working_midnight[(n, d)].Not(), 
                             is_working_midnight[(n, d+1)], 
                             off_two_days[(n, d+1)]])


# ### Hard Rule 4

# No midnights for staff in their first 6 months

# In[40]:


midnight_shifts = shifts["midnight"]
enter_date_str = doctors["enter-date"]

today = datetime.today()

for n in range(num_doctors):
    for d in range(num_days):
        for s in range(num_shifts):
            enter_date = datetime.fromisoformat(enter_date_str[n])
            if(midnight_shifts[s] == 1 and (today - enter_date).days < SIX_MONTHS):
                model.Add(doc_shifts[(n,d,s)] == 0)


# ### Hard Rule 5

# Maximum 2 midnights in a row (except for several physicians who only work midnights)

# In[41]:


night_doc = doctors["night-shift"]
midnight_shifts = shifts["midnight"]

for n in range(num_doctors):
    for d in range(num_days-2):
        sum_doc_shift = 0
        for s in range(num_shifts):
            if midnight_shifts[s] == 1 and not night_doc[n]:
                sum_doc_shift += doc_shifts[(n,d,s)] + doc_shifts[(n,d+1,s)] + doc_shifts[(n,d+2,s)]
        model.Add(sum_doc_shift <= 2)


# ### Hard Rule 7

# Certain physicians work only FT shift (0730,1530 shift)

# In[42]:


ft_doc = doctors["only-ft"]
ft_shift = shifts["fast-track"]

for n in range(num_doctors):
    for d in range(num_days):
        for s in range(num_shifts):
            if ft_doc[n] and not ft_shift[s]:
                model.Add(doc_shifts[(n,d,s)] == 0)            


# ### Hard Rule 8

# Find Day requested off from Date requested off

# In[43]:


off_date_reqs = doctors["off-day-reqs"]
# off_day_reqs[i]: off DAY requests of the i th doctor
off_day_reqs = {}

for n in range(num_doctors):
    # if there was no off days requests
    if off_date_reqs[n] != -1 and off_date_reqs[n] != "-1":
        doc_off_reqs = off_date_reqs[n].split(",")
        doc_off_reqs_list = []
        for doc_off_req in doc_off_reqs:
            doc_off_req_date = datetime.fromisoformat(doc_off_req)
            doc_off_reqs_list.append((doc_off_req_date - scheduler_start_date).days)
        off_day_reqs[n] = doc_off_reqs_list

print(off_day_reqs)


# No 2000, 2200, or midnight shift prior to day requested off

# In[44]:


start_time = shifts["start-time"]
midnight_shifts = shifts["midnight"]

for n in range(num_doctors):
    # if there was any off days requests
    if off_date_reqs[n] != -1 and off_date_reqs[n] != "-1":
        for off in off_day_reqs[n]:
            for d in range(num_days):
                for s in range(num_shifts):
                    if (midnight_shifts[s] == 1 or start_time[s] == 2000 or start_time[s] == 2200) and d == off-1:
                        print(doc_shifts[(n,d,s)] == 0)
                        model.Add(doc_shifts[(n,d,s)] == 0)


# ### Hard Rule 9

# Physicians can work the 0930 shifts or earlier prior to working on call.

# In[45]:


on_call_shifts = shifts["on-call"]
start_time = shifts["start-time"]

on_call_weight = num_shifts

for n in range(num_doctors):
    for d in range(1,num_days):     
        sum_doc_shift = 0
        for s in range(num_shifts):
            if(on_call_shifts[s] == 1):
                sum_doc_shift += on_call_weight * doc_shifts[(n,d,s)]
            elif(start_time[s] > 930):
                sum_doc_shift += doc_shifts[(n,d-1,s)]
                
#         print(sum_doc_shift)
        model.Add(sum_doc_shift <= on_call_weight)


# Doctors can work starting no earlier than 11 the day after on call.

# In[46]:


on_call_shifts = shifts["on-call"]
start_time = shifts["start-time"]

on_call_weight = num_shifts

for n in range(num_doctors):
    for d in range(num_days-1):     
        sum_doc_shift = 0
        for s in range(num_shifts):
            if(on_call_shifts[s] == 1):
                sum_doc_shift += on_call_weight * doc_shifts[(n,d,s)]
            elif(start_time[s] < 1100):
                sum_doc_shift += doc_shifts[(n,d+1,s)]
                
#         print(sum_doc_shift)
        model.Add(sum_doc_shift <= on_call_weight)


# ### Hard Rule 10

# No two doctors in the same shift on the same day

# In[47]:


for d in range(num_days):
    for s in range(num_shifts):
        model.Add(sum(doc_shifts[(n,d,s)] for n in range(num_doctors)) <= 1)


# ### Hard Rule 11

# All shifts should be taken by doctors

# In[48]:


for d in range(num_days):
    for s in range(num_shifts):
        model.Add(sum(doc_shifts[(n,d,s)] for n in range(num_doctors)) == 1)


# ## Hard Rule 12

# 2 days off after 3 to 7 days of work in a row

# In[49]:


is_working = {}

for n in range(num_doctors):
    for d in range(num_days):
        is_working[(n,d)] = model.NewBoolVar("is_working_doc%i-day%i" % (n, d))
        model.Add(is_working[(n,d)] == sum(doc_shifts[(n,d,s)] for s in range(num_shifts)))


# 2 days off after 3 consecutive working days

# In[50]:


for n in range(num_doctors):
    for d in range(num_days-5):
        days_1_3 = model.NewBoolVar("hard12_days1-3_doc%i-day%i" % (n, d))
        days_4_5 = model.NewBoolVar("hard12_days4-5_doc%i-day%i" % (n, d))
        
        days = [is_working[(n,d+d_f)] for d_f in range(5)]
        
        model.AddMultiplicationEquality(days_1_3, [days[i] for i in range(3)])
        model.AddMultiplicationEquality(days_4_5, [days[i+3] for i in range(2)])
        model.AddBoolOr([days_1_3.Not(), days_4_5.Not()])


# 2 days off after 4 consecutive working days

# In[51]:


for n in range(num_doctors):
    for d in range(num_days-6):
        days_1_4 = model.NewBoolVar("hard12_days1-4_doc%i-day%i" % (n, d))
        days_5_6 = model.NewBoolVar("hard12_days5-6_doc%i-day%i" % (n, d))
        
        days = [is_working[(n,d+d_f)] for d_f in range(6)]
        
        model.AddMultiplicationEquality(days_1_4, [days[i] for i in range(4)])
        model.AddMultiplicationEquality(days_5_6, [days[i+4] for i in range(2)])
        model.AddBoolOr([days_1_4.Not(), days_5_6.Not()])


# 2 days off after 5 consecutive working days

# In[52]:


for n in range(num_doctors):
    for d in range(num_days-7):
        days_1_5 = model.NewBoolVar("hard12_days1-5_doc%i-day%i" % (n, d))
        days_6_7 = model.NewBoolVar("hard12_days6-7_doc%i-day%i" % (n, d))
        
        days = [is_working[(n,d+d_f)] for d_f in range(7)]
        
        model.AddMultiplicationEquality(days_1_5, [days[i] for i in range(5)])
        model.AddMultiplicationEquality(days_6_7, [days[i+5] for i in range(2)])
        model.AddBoolOr([days_1_5.Not(), days_6_7.Not()])


# 2 days off after 6 consecutive working days

# In[53]:


for n in range(num_doctors):
    for d in range(num_days-8):
        days_1_6 = model.NewBoolVar("hard12_days1-6_doc%i-day%i" % (n, d))
        days_7_8 = model.NewBoolVar("hard12_days7-8_doc%i-day%i" % (n, d))
        
        days = [is_working[(n,d+d_f)] for d_f in range(8)]
        
        model.AddMultiplicationEquality(days_1_6, [days[i] for i in range(6)])
        model.AddMultiplicationEquality(days_7_8, [days[i+6] for i in range(2)])
        model.AddBoolOr([days_1_6.Not(), days_7_8.Not()])


# 2 days off after 7 consecutive working days

# In[54]:


for n in range(num_doctors):
    for d in range(num_days-9):
        days_1_7 = model.NewBoolVar("hard12_days1-7_doc%i-day%i" % (n, d))
        days_8_9 = model.NewBoolVar("hard12_days8-9_doc%i-day%i" % (n, d))
        
        days = [is_working[(n,d+d_f)] for d_f in range(9)]
        
        model.AddMultiplicationEquality(days_1_7, [days[i] for i in range(7)])
        model.AddMultiplicationEquality(days_8_9, [days[i+7] for i in range(2)])
        model.AddBoolOr([days_1_7.Not(), days_8_9.Not()])


# ## Objective Function

# In[55]:


soft = [0 for i in range(10)]
obj = [0 for i in range(14)]


# ## Soft Rules

# ### Soft Rule 1

# General principle avoid shift times changing too much day to day

# Shifts should have same start time to 2.5 hours later compared to previous shift (the 2 hours later can be relaxed to 3,4 perhaps)

# In[56]:


# Find distant shifts (more than 2h:30m difference in start_time)
start_time = shifts["start-time"]
distant_shifts = []
for s1 in range(num_shifts):
    for s2 in range(num_shifts):
        if abs(start_time[s1] - start_time[s2]) > 230:
            distant_shifts.append((s1, s2))
print(distant_shifts)


# In[57]:


soft[1] = 0
tmp_soft1_var_1, tmp_soft1_var_2 = {}, {}
for n in range(num_doctors):
    for d in range(num_days - 1):
        for s_s in distant_shifts:
            tmp_soft1_var_1[(n,d,s_s[0])] = model.NewIntVar(0, 2, "soft1_minus_doc%i-day%i-shift%i" % (n, d, s_s[0]))
            tmp_soft1_var_2[(n,d,s_s[0])] = model.NewBoolVar("soft1_abs_doc%i-day%i-shift%i" % (n, d, s_s[0]))
            
            model.AddMultiplicationEquality(tmp_soft1_var_1[(n,d,s_s[0])], [doc_shifts[(n, d, s_s[0])], doc_shifts[(n, d+1, s_s[1])]])
#             model.AddAbsEquality(tmp_soft1_var_2[(n,d,s_s[0])], tmp_soft1_var_1[(n,d,s_s[0])])
            soft[1] += tmp_soft1_var_1[(n,d,s_s[0])]
soft[1] *= -1


# ### Soft Rule 2

# 3 days off after midnight. 4 even better

# In[58]:


soft[2] = 0

is_not_off_after_midnight = {}

# list of midnight-shift indexes
midnight_shifts = shifts.index[shifts['midnight'] == 1].tolist()

for n in range(num_doctors):
    for d in range(num_days-3):
        for s in midnight_shifts:
            is_working_in_three_days = model.NewBoolVar("is_working_in_next_three_days(n%i,d%i)" % (n, d))
            model.Add(is_working_in_three_days == (is_working[(n,d+1)] + is_working[(n,d+2)]))
            tmp = [doc_shifts[(n,d,s)], is_working_in_three_days]
            is_not_off_after_midnight[(n,d)] = model.NewBoolVar("daysoff_after_midnight(n%i,d%i)" % (n, d))
            model.AddMultiplicationEquality(is_not_off_after_midnight[(n,d)], tmp)
            soft[2] += is_not_off_after_midnight[(n,d)]
soft[2] *= -1


# ### Soft Rule 3

# Max 1 midnight in a row

# In[59]:


is_working_midnight = {}

midnight_shifts = shifts["midnight"]
for n in range(num_doctors):
    for d in range(num_days):
        for s in range(num_shifts):
            if midnight_shifts[s] == 1:
                is_working_midnight[(n, d)] = model.NewBoolVar("is_working_midnight_doc%i-day%i" % (n, d))
                model.Add(is_working_midnight[(n, d)] == doc_shifts[(n, d, s)])


# In[60]:


is_working_two_midnight = {}

if 1 in midnight_shifts.values:
    for n in range(num_doctors):
        for d in range(num_days - 1):
            for s in range(num_shifts):
                is_working_two_midnight[(n, d)] = model.NewBoolVar("is_working_two_midnight_doc%i-day%i" % (n, d))
                model.AddMultiplicationEquality(is_working_two_midnight[(n, d)], 
                                                [is_working_midnight[(n,d)], is_working_midnight[(n, d+1)]])
    soft[3] = sum(is_working_two_midnight[(n, d)] for n in range(num_doctors) for d in range(num_days - 1))

    # we want to minimize this constraint
    soft[3] *= -1


# ### Soft Rule 4

# 3 days off when transitioning from late shift to day shift

# In[61]:


soft[4] = 0

is_transitioning_late_2_day = {}
is_working_next_3_days = {}

# get the day shifts
day_shifts = shifts.index[shifts['day-shift'] == 1].tolist()

# get the late shifts
late_shifts = shifts.index[shifts['late-shift'] == 1].tolist()

for n in range(num_doctors):
    for d in range(num_days-4):
        for s_day in day_shifts:
            for s_night in late_shifts:
                # is_transitioning_late_2_day = late_shift[today] * day_shift[tomorrow]
                is_transitioning_late_2_day[(n,d)] = model.NewBoolVar("is_transitioning_late_2_day%i-day%i" % (n, d))
                is_working_late_shift_today = doc_shifts[(n,d,s_night)]
                is_working_day_shift_tomorrow = doc_shifts[(n,d+1,s_day)]
                
                model.AddMultiplicationEquality(is_transitioning_late_2_day[(n,d)], 
                                                [is_working_late_shift_today, is_working_day_shift_tomorrow])
                
                # is_working_next_3_days = is_working[tomorrow+1] * is_working[tomorrow+2] * is_working[tomorrow+3]
                is_working_next_3_days[(n,d)] = model.NewBoolVar("is_working_next_3_days%i-day%i" % (n, d))
                model.Add(is_working_next_3_days[(n,d)] == 
                                                (is_working[(n,d+2)] + is_working[(n,d+3)]) )
                
                is_not_soft4_satisfied = model.NewBoolVar("is_not_soft_rule_4_satisfied%i-day%i" % (n, d))
                model.AddMultiplicationEquality(is_not_soft4_satisfied, 
                                                [is_transitioning_late_2_day[(n,d)], is_working_next_3_days[(n,d)]])
                soft[4] += is_not_soft4_satisfied

soft[4] *= -1


# ### Soft Rule 5

# In[62]:


soft[5] = 0

is_transitioning_late_2_afternoon = {}
is_working_next_2_days = {}

# get the afternoon shifts
afternoon_shifts = shifts.index[shifts['afternoon-shift'] == 1].tolist()

# get the late shifts
late_shifts = shifts.index[shifts['late-shift'] == 1].tolist()

for n in range(num_doctors):
    for d in range(num_days-3):
        for s_afternoon in afternoon_shifts:
            for s_night in late_shifts:
                # is_transitioning_late_2_afternoon = late_shift[today] * afternoon_shift[tomorrow]
                is_transitioning_late_2_afternoon[(n,d)] = model.NewBoolVar("is_transitioning_late_2_afternoon%i-day%i" % (n, d))
                is_working_late_shift_today = doc_shifts[(n,d,s_night)]
                is_working_afternoon_shift_tomorrow = doc_shifts[(n,d+1,s_afternoon)]
                
                model.AddMultiplicationEquality(is_transitioning_late_2_afternoon[(n,d)], 
                                                [is_working_late_shift_today, is_working_afternoon_shift_tomorrow])
                
                # is_working_next_2_days = is_working[tomorrow+2] * is_working[tomorrow+3]
                is_working_next_2_days[(n,d)] = model.NewBoolVar("is_working_next_2_days%i-day%i" % (n, d))
                model.Add(is_working_next_2_days[(n,d)] == 
                                                (is_working[(n,d+2)] + is_working[(n,d+3)]) )
                
                is_not_soft5_satisfied = model.NewBoolVar("is_not_soft_rule_5_satisfied%i-day%i" % (n, d))
                model.AddMultiplicationEquality(is_not_soft5_satisfied, 
                                                [is_transitioning_late_2_afternoon[(n,d)], is_working_next_2_days[(n,d)]])
                soft[5] += is_not_soft5_satisfied

soft[5] *= -1


# ### Soft Rule 6

# 3 late shifts in a row maximum - late shifts are 1800, 2000, 2200.

# In[63]:


is_working_late = {}

late_shifts = shifts.index[shifts['late-shift'] == 1].tolist()

# if there is any late shifts
if(late_shifts):
    for n in range(num_doctors):
        for d in range(num_days):
            late_shifts_sum = 0
            for s in late_shifts:
                is_working_late[(n,d)] = model.NewBoolVar("is_working_late_doc%i-day%i" % (n, d))
                late_shifts_sum = doc_shifts[(n,d,s)]
            model.Add(is_working_late[(n, d)] == late_shifts_sum)


# In[64]:


soft[6] = 0

is_working_late_4days = {}

# if there is any late shifts
if(late_shifts):
    for n in range(num_doctors):
        for d in range(num_days - 4 + 1):
            is_working_var_list = [is_working_late[(n,d+d_f)] for d_f in range(4)]
            is_working_late_4days[(n,d)] = model.NewBoolVar("is_working_late_4days(%i,%i)" % (n, d))

            model.AddMultiplicationEquality(is_working_late_4days[(n,d)], is_working_var_list)
            soft[6] += is_working_late_4days[(n,d)]

soft[6] *= -1


# ### Soft Rule 7

# 5 late shifts in two weeks maximum

# In[65]:


# soft[7] = 0

# midnight_shifts = shifts.index[shifts['midnight'] == 1].tolist()

# for n in range(num_doctors):
#     for d in range(num_days-14+1):
#         sum(doc_shifts[(n,d+d_f,s_mn) for s in range])


# ### Soft Rule 8

# Work maximum of 5 days in a row

# In[66]:


soft[8] = 0

for n in range(num_doctors):
    for d in range(num_days - 5 + 1):
        is_working_var_list = [is_working[(n,d+d_f)] for d_f in range(5)]
        is_working_days_1_5 = model.NewBoolVar("is_working_days_1_5(%i,%i)" % (n, d))
        
        model.AddMultiplicationEquality(is_working_days_1_5, is_working_var_list)
        soft[8] += is_working_days_1_5

soft[8] *= -1


# ### Soft Rule 9

# Avoid FT shifts (0730,1530) on consecutive days

# In[67]:


# consecutive_days = []
# soft[9] = 0

# consec_ft = {}

# # list of ft-shift indexes
# ft_shifts = shifts.index[shifts['fast-track'] == 1].tolist()

# for n in  range(num_doctors):
#     for d in range(num_days-1):
#         for s1 in ft_shifts:
#             for s2 in ft_shifts:
#                 if s1 != s2:
#                     consec_ft[(n,d,s1,s2)] = model.NewBoolVar("consec_ft(%i,%i,%i,%i)" % (n, d, s1, s2))
#                     model.AddMultiplicationEquality(consec_ft[(n,d,s1,s2)], [doc_shifts[(n,d,s1)], doc_shifts[(n,d+1,s2)]])
# # print(consec_ft.get((n,d,s1,s2), 0) for n in range(num_doctors) for d in range(num_days) for s1 in ft_shifts for s2 in ft_shifts)
# soft[9] = sum(consec_ft.get((n,d,s1,s2), 0) for n in range(num_doctors) for d in range(num_days) for s1 in ft_shifts for s2 in ft_shifts)

# soft[9] *= -1


# ### Objective 1

# Equalize weekends

# In[68]:


obj[1] = 0

sum_doc_weekend_worked = {}

for n in range(num_doctors):
    sum_doc_weekend_worked[n] = model.NewIntVar(0, num_days, "doc%i_sum_weekend_shift" % n)
    sum_weekend_worked = 0
    for d in range(num_days):
        if is_weekend(d):   
            sum_weekend_worked += is_working[(n,d)]
    model.Add(sum_doc_weekend_worked[n] == sum_weekend_worked)

# find the max number a doctor is working on weekends
max_sum_weekend_shift = model.NewIntVar(0, num_days, "doc%i_max_weekend_sum_shift" % n)
model.AddMaxEquality(max_sum_weekend_shift, [sum_doc_weekend_worked[n] for n in range(num_doctors)])

# find the min number a doctor is working on weekends
min_sum_weekend_shift = model.NewIntVar(0, num_days, "doc%i_min_weekend_sum_shift" % n)
model.AddMinEquality(min_sum_weekend_shift, [sum_doc_weekend_worked[n] for n in range(num_doctors)])

# minimize the difference of the max and min numbers doctors scheduled for weekend shifts
# minus is for maximizing the negation
obj[1] = -(max_sum_weekend_shift - min_sum_weekend_shift)


# ### Objective 2

# Minimize weekends working only 1 day (‘split weekends’)

# In[69]:


# obj[2] = 0
# sat_plus_sun = {}
# split_weekends = {}

# for n in range(num_doctors):
#     for d in range(num_days - 1):
#         if is_saturday(d):
#             sat_plus_sun[(n,d)] = model.NewIntVar(0, 2, "sat_plus_sun%i_d%i" % (n,d))
#             split_weekends[(n,d)] = model.NewIntVar(0, 2, "splitweekend_n%i_d%i" % (n,d))
#             model.Add(sat_plus_sun[(n,d)] == (is_working[(n,d)] + is_working[(n,d+1)]))
#             # if split_weekends = 1 -> doc n is working either on saturday or sunday
#             # the sum of (working on saturday) and (working on sunday) is better not be odd (sum % 2 != 1)
#             model.AddModuloEquality(split_weekends[(n,d)], sat_plus_sun[(n,d)], 2)


# # if split_weekend with the key not found --> use default value of 0
# obj[2] = sum(split_weekends.get((n,d), 0) for n in range(num_doctors) for d in range(num_days))
# obj[2] *= -1


# ### Objective 3

# Equalize holidays worked

# In[70]:


obj[3] = 0

sum_doc_holiday_worked = {}

holiday_days = []

for h in holidays_ontario_canada_date:
    holiday_day = date_to_day(h)
#     print(holiday_day)
    # append only if the holiday is within the scheduler period
    if holiday_day > 0 and holiday_day < num_days:
        holiday_days.append(holiday_day)

print(holiday_days)
for n in range(num_doctors):
    sum_doc_holiday_worked[n] = model.NewIntVar(0, num_days, "doc%i_sum_holiday_worked" % n)
    sum_holiday_worked = 0
    for hd in holiday_days:
        sum_holiday_worked += is_working[(n,hd)]
    model.Add(sum_doc_holiday_worked[n] == sum_holiday_worked)

# find the max number a doctor is working on holidays
max_sum_holiday_worked = model.NewIntVar(0, num_days, "doc%i_max_holiday_sum_shift" % n)
model.AddMaxEquality(max_sum_holiday_worked, [sum_doc_holiday_worked[n] for n in range(num_doctors)])

# find the min number a doctor is working on holidays
min_sum_holiday_worked = model.NewIntVar(0, num_days, "doc%i_min_holiday_sum_shift" % n)
model.AddMinEquality(min_sum_holiday_worked, [sum_doc_holiday_worked[n] for n in range(num_doctors)])

# minimize the difference of the max and min numbers doctors scheduled for holiday shifts
# minus is for maximizing the negation
obj[3] = -(max_sum_holiday_worked - min_sum_holiday_worked)


# ### Objective 5

# Avoid scheduling people shifts that end after 5 pm Friday on weekends they have off

# In[71]:


obj[5] = 0

start_time = shifts["start-time"]
midnight_shifts = shifts["midnight"]

for n in range(num_doctors):
    # if there was any off days requests
    if off_date_reqs[n] != -1 and off_date_reqs[n] != "-1":
        for off in off_day_reqs[n]:
            for d in range(num_days):
                for s in range(num_shifts):
                    if (start_time[s] > 1700) and d == off:
                        obj[5] += doc_shifts[(n,d,s)]
obj[5] *= -1


# In[72]:


# print(obj[5])


# ### Objective 7

# Equalize day shifts (0700 - 1200 start time)

# In[73]:


shift_index = {}
    
sum_doc_day_shifts_var = {}
for n in range(num_doctors):
    sum_doc_day_shifts = 0
    day_shifts = get_day_shifts()
    for s in day_shifts:
        sum_doc_day_shifts_var[n] = model.NewIntVar(0, num_days, "doc%i_sum_day_shift" % n)
        for d in range(num_days):
            sum_doc_day_shifts += doc_shifts[(n,d,s)]
        model.Add(sum_doc_day_shifts_var[n] == sum_doc_day_shifts)

# find the max number a doctor is scheduled for day shifts
max_sum_shift = model.NewIntVar(0, num_days, "doc%i_max_sum_day_shift" % n)
model.AddMaxEquality(max_sum_shift, [sum_doc_day_shifts_var[n] for n in range(num_doctors)])

# find the min number a doctor is scheduled for day shifts
min_sum_shift = model.NewIntVar(0, num_days, "doc%i_min_sum_day_shift" % n)
model.AddMinEquality(min_sum_shift, [sum_doc_day_shifts_var[n] for n in range(num_doctors)])

# minimize the difference of the max and min numbers doctors scheduled for day shifts
# minus is for maximizing the negation
obj[7] = -(max_sum_shift - min_sum_shift)


# ### Objective 9

# Equalize 2200 shifts

# In[74]:


shift_index = {}

# if shift 2200 exists
for start_time in shifts['start-time']:
    if start_time == 2200:

        shift_index[2200] = shifts.loc[shifts['start-time'] == 2200].index.values[0]
        # find sum shift 2200 for each doctor
        doc_sum_shifts = {}
        for n in range(num_doctors):
            sum_doc_shifts = 0
            doc_sum_shifts[(n, shift_index[2200])] = model.NewIntVar(0, num_days, "doc%i_sum_shift%i" % (n, shift_index[2200]))
            for d in range(num_days):
                sum_doc_shifts += doc_shifts[(n,d,shift_index[2200])]
            model.Add(doc_sum_shifts[(n,shift_index[2200])] == sum_doc_shifts)


        # find the max number a doctor is scheduled shift 2200
        max_sum_shift = model.NewIntVar(0, num_days, "doc%i_max_sum_shift%i" % (n, shift_index[2200]))
        model.AddMaxEquality(max_sum_shift, [doc_sum_shifts[(n,shift_index[2200])] for n in range(num_doctors)])

        # find the min number a doctor is scheduled shift 2200
        min_sum_shift = model.NewIntVar(0, num_days, "doc%i_min_sum_shift%i" % (n, shift_index[2200]))
        model.AddMinEquality(min_sum_shift, [doc_sum_shifts[(n,shift_index[2200])] for n in range(num_doctors)])

        # minimize the difference of the max and min numbers doctors scheduled for shift 2200
        # minus is for maximizing the negation
        obj[9] = -(max_sum_shift - min_sum_shift)


# ### Objective 10

# Equalize 2000 shifts

# In[75]:


shift_index = {}

shift_name_num = 2000
# if shift 2000 exists
for start_time in shifts['start-time']:
    if start_time == shift_name_num:

        shift_index[shift_name_num] = shifts.loc[shifts['start-time'] == shift_name_num].index.values[0]
        # find sum shift 2200 for each doctor
        doc_sum_shifts = {}
        for n in range(num_doctors):
            sum_doc_shifts = 0
            doc_sum_shifts[(n, shift_index[shift_name_num])] = model.NewIntVar(0, num_days, "doc%i_sum_shift%i" % (n, shift_index[shift_name_num]))
            for d in range(num_days):
                sum_doc_shifts += doc_shifts[(n,d,shift_index[shift_name_num])]
            model.Add(doc_sum_shifts[(n,shift_index[shift_name_num])] == sum_doc_shifts)


        # find the max number a doctor is scheduled shift 2000
        max_sum_shift = model.NewIntVar(0, num_days, "doc%i_max_sum_shift%i" % (n, shift_index[shift_name_num]))
        model.AddMaxEquality(max_sum_shift, [doc_sum_shifts[(n,shift_index[shift_name_num])] for n in range(num_doctors)])

        # find the min number a doctor is scheduled shift 2000
        min_sum_shift = model.NewIntVar(0, num_days, "doc%i_min_sum_shift%i" % (n, shift_index[shift_name_num]))
        model.AddMinEquality(min_sum_shift, [doc_sum_shifts[(n,shift_index[shift_name_num])] for n in range(num_doctors)])

        # minimize the difference of the max and min numbers doctors scheduled for shift 2000
        # minus is for maximizing the negation
        obj[10] = -(max_sum_shift - min_sum_shift)


# ### Objective 11

# Equalize weekdays (same number of shifts on M,T,W,Th,F

# In[76]:


# obj[11] = 0

# sum_doc_weekday_worked = {}

# for n in range(num_doctors):
#     sum_doc_weekday_worked[n] = model.NewIntVar(0, num_days, "doc%i_sum_weekday_shift" % n)
#     sum_weekday_worked = 0
#     for d in range(num_days):
#         if is_weekday(d):   
#             sum_weekday_worked += is_working[(n,d)]
#     model.Add(sum_doc_weekday_worked[n] == sum_weekday_worked)

# # find the max number a doctor is working on weekdays
# max_sum_weekday_shift = model.NewIntVar(0, num_days, "doc%i_max_weekday_sum_shift" % n)
# model.AddMaxEquality(max_sum_weekday_shift, [sum_doc_weekday_worked[n] for n in range(num_doctors)])

# # find the min number a doctor is working on weekdays
# min_sum_weekday_shift = model.NewIntVar(0, num_days, "doc%i_min_weekday_sum_shift" % n)
# model.AddMinEquality(min_sum_weekday_shift, [sum_doc_weekday_worked[n] for n in range(num_doctors)])

# # minimize the difference of the max and min numbers doctors scheduled for weekday shifts
# # minus is for maximizing the negation
# obj[11] = -(max_sum_weekday_shift - min_sum_weekday_shift)


# ### Objective 13

# Avoid working more than 3 days in row

# In[77]:


# is_working = {}
# for n in range(num_doctors):
#     for d in range(num_days):
#         is_working[(n,d)] = model.NewBoolVar("doc%i_working_day%i" % (n, d))
#         model.Add(is_working[(n,d)] == sum(doc_shifts[(n,d,s)] for s in range(num_shifts)))


# In[78]:


# four_day_iterator = {}
# obj[13] = 0
# for n in range(num_doctors):
#     for d in range(num_days-3):
#         four_day_iterator[(n,d)] = model.NewBoolVar("doc%i_day%i-%i" % (n, d, d+3))          
#         model.AddMultiplicationEquality(four_day_iterator[(n,d)], [is_working[(n,d+0)], is_working[(n,d+1)], is_working[(n,d+2)], is_working[(n,d+3)]])
#         obj[13] += four_day_iterator[(n,d)]

# obj[13] *= -1


# ### Maximize the Objective Function

# Using variant coefficients based on each objective function and its prriority

# In[79]:


model.Maximize(sum(soft) + sum(obj))


# ## Results

# ### Display Doctors

# In[80]:


doctors


# ### Display Shifts

# In[81]:


shifts


# ### Calling Solver and Solution Printer

# In[82]:


stat_file = open("stat.txt", "w")


# In[83]:


def print_stat(stat):
    print(stat)
    stat_file.write("%s\n" % stat)


# In[84]:


solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 36000.0
start_exec_time = time.time()
status = solver.Solve(model)
end_exec_time = time.time()

print_stat("Execution Time: %f" % (end_exec_time - start_exec_time))


# In[85]:


if status == cp_model.FEASIBLE:
    print_stat("Status: Feasible")
elif status == cp_model.OPTIMAL:
    print_stat("Status: Optimal")
elif status == cp_model.UNKNOWN:
    print_stat("Status: Unknown")
    stat_file.close()
elif status == cp_model.INFEASIBLE:
    print_stat("Status: Infeasible")
    stat_file.close()


# In[86]:


print_stat("---------------------")


# ## Print solutions

# ### Print text

# In[87]:


output_file = open("schedules.txt", "w", encoding="utf-8")


# In[88]:


if status != cp_model.INFEASIBLE and status != cp_model.UNKNOWN:
    day_list = []
    for s in range(num_shifts):                                                                                                                                                                               
        shift_list = []
        shift_list.append("Shift " + shifts["name"][s])
        for d in range(num_days):
            doc = -1
            for n in range(num_doctors):
                if solver.Value(doc_shifts[(n,d,s)]):
                    doc = n
            shift_list.append(doctors["firstname"][doc] + " " + doctors["lastname"][doc])
        day_list.append(shift_list)
    
    headers = [(scheduler_start_date + timedelta(days=d)).strftime("%d %b, %Y") for d in range(num_days)]
    table = tabulate(day_list, headers, tablefmt="fancy_grid")
    
    output_file.write(table)
#     print(table)
#     print("------------------------------------------------------------------")


# In[89]:


output_file.close()


# #### Print CSV

# In[90]:


output_file = open("schedules.csv", "w", encoding="utf-8")
output_csv_file = csv.writer(output_file)


# In[91]:


if status != cp_model.INFEASIBLE and status != cp_model.UNKNOWN:
    headers = [(scheduler_start_date + timedelta(days=d)).strftime("%d %b, %Y") for d in range(num_days)]
    day_list = []
    for s in range(num_shifts):                                                                                                                                                                               
        shift_list = []
        shift_list.append("Shift " + shifts["name"][s])
        for d in range(num_days):
            doc = -1
            for n in range(num_doctors):
                if solver.Value(doc_shifts[(n,d,s)]):
                    doc = n
            shift_list.append(doctors["firstname"][doc] + " " + doctors["lastname"][doc])
        day_list.append(shift_list)
    
    headers = [(scheduler_start_date + timedelta(days=d)).strftime("%d %b, %Y") for d in range(num_days)]
    headers.insert(0, "")
#     table = tabulate(day_list, headers, tablefmt="fancy_grid")
    
    output_csv_file.writerows([headers])
    output_csv_file.writerows(day_list)
#     print(table)
#     print("------------------------------------------------------------------")


# In[92]:


output_file.close()


# In[93]:


# solution_printer = DoctorsPartialSolutionPrinter(doc_shifts, 1)
# solver = cp_model.CpSolver()
# status = solver.SearchForAllSolutions(model, solution_printer)


# In[94]:


sum_soft = 0
for i, s in enumerate(soft):
    s_val = solver.Value(s)
    sum_soft += s_val
    print_stat("soft%d value: %d" % (i, s_val))

print_stat("---------------------")
print_stat("Sum soft value: %d" % sum_soft)


# In[95]:


sum_obj = 0
for i, o in enumerate(obj):
    o_val = solver.Value(o)
    sum_obj += o_val
    print_stat("obj%d value: %d" % (i, o_val))

print_stat("---------------------")
print_stat("Sum obj value: %d" % sum_obj)


# In[96]:


stat_file.close()


# In[97]:


get_shift_status(solver, doc_shifts, 0, 0)
get_shift_status(solver, doc_shifts, 0, 1)
get_shift_status(solver, doc_shifts, 0, 2)
get_shift_status(solver, doc_shifts, 1, 0)
get_shift_status(solver, doc_shifts, 2, 0)


# ### Save report files

# do not show figures by default

# In[98]:


plt.ioff()


# Create sum working days report file

# In[99]:


path_reports = "reports/"
if not os.path.exists(path_reports):
    os.makedirs(path_reports)


# In[100]:


x = [n for n in range(num_doctors)]
y = [get_sum_working_days(solver, doc_shifts, n) for n in range(num_doctors)]

f = plt.figure("sum_working_days")
plt.bar(x, y)
y_ticks = np.arange(0, max(y) + 1, 1)
plt.yticks(y_ticks)
plt.xlabel("doctor ID")
plt.ylabel("sum days working")
plt.suptitle('Sum days doctors working')

path_file = path_reports + 'sum_working_days.png'
f.savefig(path_file , dpi=100)
plt.close(f)
# f.show()


# Create sum shifts report files

# In[101]:


for s in range(num_shifts):
    x = [n for n in range(num_doctors)]
    y = [get_sum_working_shift(solver, doc_shifts, n, s) for n in range(num_doctors)]

    shift_name = get_shift_name(s)
    f = plt.figure("sum_working_shift%s" % shift_name)
    plt.bar(x, y)
    y_ticks = np.arange(0, max(y) + 1, 1)
    plt.yticks(y_ticks)
    plt.xlabel("doctor ID")
    plt.ylabel("sum days %s" % shift_name)
    plt.suptitle('Sum shift %s working' % shift_name)
    plt.close(f)

    path_file = path_reports + ('sum_shift_%s.png' % shift_name)
    f.savefig(path_file, dpi=100)


# Create sum holidays working

# In[102]:


x = [n for n in range(num_doctors)]
y = [get_sum_working_holidays(solver, doc_shifts, holiday_days, n) for n in range(num_doctors)]

shift_name = get_shift_name(s)
f = plt.figure("sum_working_holidays")
plt.bar(x, y)
y_ticks = np.arange(0, max(y) + 1, 1)
plt.yticks(y_ticks)
plt.xlabel("doctor ID")
plt.ylabel("sum days")
plt.suptitle('Sum working holidays')
plt.close(f)

path_file = path_reports + ('sum_working_holidays.png')
f.savefig(path_file, dpi=100)


# Create sum day shifts report files

# In[103]:


x = [n for n in range(num_doctors)]
y = [get_sum_day_shifts(solver, doc_shifts, n) for n in range(num_doctors)]

f = plt.figure("Sum day shifts")
plt.bar(x, y)
y_ticks = np.arange(0, max(y) + 1, 1)
plt.yticks(y_ticks)
plt.xlabel("doctor ID")
plt.ylabel("sum days")
plt.suptitle('Sum day shifts')
plt.close(f)

path_file = path_reports + ('sum_day_shifts.png')
f.savefig(path_file, dpi=100)


# In[ ]:





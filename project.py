# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 14:06:19 2019

@author: Kevin, Adonis, Cedric, Shenghua
"""

import pandas as pd
import numpy as np
import statistics
import statsmodels.api as sm
import math
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#Goal: Run regressions to see what influences the number of turns of a chess match

#Comparing number of moves and ranking difference
#-----------------------------------------------------------------------------

df_chess_K = pd.read_csv("chess.csv", usecols=["rated", "turns", "white_rating", "black_rating"])
#Query 1: subsetting only the matches that were "ranked," i.e. playing for ELO score consequences.
df_chess_final_K = df_chess_K.query("rated == True")

df_chess_final_K['difference'] = df_chess_final_K['white_rating'] - df_chess_final_K['black_rating']
df_chess_final_K["difference"] = abs(df_chess_final_K["difference"])


statistics.median(df_chess_final_K['turns'])
#Median number of turns is 57. this is equal to half moves



endog2 = df_chess_final_K['turns']
exog2 = sm.add_constant(df_chess_final_K["difference"])

mod = sm.OLS(endog2, exog2).fit()
#mod.summary()
#Expected number of turns for two evenly ranked opponents is 66. An increase 
#In raking difference by 100 corresponds to a 2.60 move difference.
#------------------------------------------------------------------------------

#Comparing number of moves and player skill level
#------------------------------------------------------------------------------
df_chess_final_K['average'] = (df_chess_final_K['white_rating'] + df_chess_final_K['black_rating'])/2

endog3 = df_chess_final_K['turns']
exog3 = sm.add_constant(df_chess_final_K[["difference", "average"]])


mod = sm.OLS(endog3, exog3).fit()
#mod.summary()
#Increase in average rating by 100 corresponds to 2.19 increase in # of moves. 
#------------------------------------------------------------------------------

#Plots of these two relationships, commented out for program running
#------------------------------------------------------------------------------

#plt.scatter(df_chess_final_K["difference"], df_chess_final_K["turns"], alpha = 0.5)
#plt.title('Rating Difference vs Number of Turns')
#plt.xlabel('Rating Difference')
#plt.ylabel('Number of Turns')
#plt.show()
#
#plt.scatter(df_chess_final_K["average"], df_chess_final_K["turns"], alpha = 0.5)
#plt.title('Rating Average vs Number of Turns')
#plt.xlabel('Rating Average')
#plt.ylabel('Number of Turns')
#plt.show()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Main linear regression model predicting win probability for white
#------------------------------------------------------------------------------
df_chess_2_K = pd.read_csv("chess.csv", usecols=["rated", "winner", "white_rating", "black_rating"])
df_chess_final_2_K = df_chess_2_K.query("rated == True")

df_chess_final_2_K['difference'] = df_chess_final_2_K['white_rating'] - df_chess_final_2_K['black_rating']

df_chess_final_2_K['w_win'] = 0
df_chess_final_2_K.loc[(df_chess_final_2_K['winner'] == "white"), 'w_win'] = 1

#Query 2: Seeing what percent of matches white won
df_chess_final_K_check = df_chess_final_2_K.query('winner == "white"')
#49.84% of ranked games won by white

endog4 = df_chess_final_2_K['w_win']
exog4 = sm.add_constant(df_chess_final_2_K[["difference", "white_rating"]])
#plt.scatter(df_chess_final_2_K["difference"],df_chess_final_2_K["w_win"])



logit_model2 = sm.Logit(endog4, (exog4)).fit()
mod_coef = sm.Logit(endog4, exog4).fit().params.values
#print(logit_model2.summary())

#The variable "white_rating" shows that lower ratings corresponds to a higher chance
#of winning, with p around 6%. We decided to leave this in the model because it
#makes sense, given the known fact that more skilled players draw more. 

#----------------------------------------------------------------------------------------
 
#Checking to see if a higher rating for white leads to greater chance for win or draw
#------------------------------------------------------------------------------------
df_chess_final_2_K['w_win'] = 1
df_chess_final_2_K.loc[(df_chess_final_2_K['winner'] == "black"), 'w_win'] = 0



endog4 = df_chess_final_2_K['w_win']
exog4 = sm.add_constant(df_chess_final_2_K[["difference", "white_rating"]])
#plt.scatter(df_chess_final_2["difference"],df_chess_final_2["w_win"])



logit_model = sm.Logit(endog4, sm.add_constant(exog4)).fit()
#print(logit_model.summary())
#P-value is 0.139, meaning there is not quite proof that playing as white is a 
#bigger advantage for better players

#-----------------------------------------------------------------------------

#Are there more draws for higher rated players?
df_chess_final_2_K['w_win'] = 0
df_chess_final_2_K.loc[(df_chess_final_2_K['winner'] == "draw"), 'w_win'] = 1


df_chess_final_2_K["difference"] = abs(df_chess_final_2_K["difference"])
endog4 = df_chess_final_2_K['w_win']
exog4 = sm.add_constant(df_chess_final_2_K[["difference", "white_rating"]])
#plt.scatter(df_chess_final_2["difference"],df_chess_final_2["w_win"])



logit_model = sm.Logit(endog4, sm.add_constant(exog4)).fit()
mod_coef2 = sm.Logit(endog4, exog4).fit().params.values
#print(logit_model.summary())
#There is evidence right here, with a z-value of 3.994, that there is an increased
#number of draws for increased player ratings.
#-----------------------------------------------------------------------------------

print("Hello, we will tell you your chance of winning a chess match")
elo1 = input("Please enter the white side's ELO ranking : ") 
elo2 = input("Now enter the black side's ELO rating : ")


def chessprobwin(x, y):
    beta = mod_coef[0] + (mod_coef[1]*(int(elo1)-int(elo2))) + (mod_coef[2]*int(elo1))
    prob = math.exp(beta) / (1+ (math.exp(beta)))
    return prob

def chessprobdraw(x, y):
    diff = abs(int(elo1) - int(elo2))
    beta = mod_coef2[0] + (mod_coef2[1]*(diff)) + (mod_coef2[2]*int(elo1))
    prob = math.exp(beta) / (1+ (math.exp(beta)))
    return prob

def outputprob():
    print("White's probability of winning is: ")
    print(round(chessprobwin(elo1,elo2), 3))

    print("The probability of a draw is: ")
    print(round(chessprobdraw(elo1,elo2), 3))

    print("Black's probability of winning is: ")
    xzx = (1 - (chessprobwin(elo1, elo2)+(chessprobdraw(elo1, elo2))))
    print(round((xzx),3))
    
outputprob()



#----------------------------------------------------------------------------



#A Further analysis of the probability of winning a match, including a plot



df_chess_s = pd.read_csv("chess.csv", usecols=["rated", "winner", "white_rating", "black_rating"])
df_chess_s = df_chess_s.query("rated == True")
df_chess_s['difference'] = df_chess_s['black_rating'] - df_chess_s['white_rating']

# add theoretical white win rate to the data frame
df_chess_s['w_winRate'] = 1 / (1+ np.power(10, df_chess_s['difference'] /400))

# a function to set a new column in the data frame
def transform(df, x, y, string):
    df[x] = 0
    df.loc[(df[y] == string), x] = 1

# use the transform function to get a new column in the data frame shown the wins of white
transform(df_chess_s, 'w_win', 'winner', 'white')

# plot logistic regression model and win rate of white in the graph
X = df_chess_s["difference"].values.reshape(-1,1)
Y = df_chess_s["w_win"].values.reshape(-1,1)
LogR = LogisticRegression()
LogR.fit(X ,np.ravel(Y.astype(int)))
#plt.scatter(X, Y,alpha = 0.5, label = 'sample')
#plt.scatter(X,LogR.predict_proba(X)[:,1], c = 'red', alpha = 0.5, label = 'Practical')
#plt.scatter(X, df_chess_s['w_winRate'], alpha = 0.5, c = 'green', label = 'Theoretical')
#plt.title('Win Rate of White vs Difference between Elo Rating')
#plt.xlabel('Black Elo Rating - White Elo Rating')
#plt.ylabel('Win Rate of White')
#plt.legend(loc=1)
#plt.show()

# Set up the logistic regression model and get the summary
endog4 = df_chess_s['w_win']
exog4 = sm.add_constant(df_chess_s[["difference", "white_rating"]])
logit_model = sm.Logit(endog4, (exog4)).fit()
#print(logit_model.summary())

#-----------------------------------------------------------------------------



#Exploring HOW a match ends: draw, resign, checkmate

 

#Idea: Lower Elo -> more checkmate, worse players do not foresee loss and resign
#Y variable is 0-1 where a 1 is a checkmate. 2 x variables here: abs(elo difference) and average(elo)

#Higher Elo -> more resigns, good players foresee a loss
#Y variable is 0-1 where a 1 is a resign. 2 x variables here: abs(elo difference) and average(elo)


#read in csv
df_chess_C = pd.read_csv("chess.csv", usecols=["rated","winner", "white_rating", "black_rating", "victory_status"])

#Filtering out all of the rated rows that are false and filtering out out of time matches 
#and victory_status != "outoftime"
df_chess_C_rated = df_chess_C.query('rated == True')

#first predictor-abs of difference in rating pred1
df_chess_C_rated['difference_C'] = df_chess_C_rated['white_rating'] - df_chess_C_rated['black_rating']

df_chess_C_rated["difference_C"] = abs(df_chess_C_rated["difference_C"])

#second predictor-average rating between two players pred2 
df_chess_C_rated['average_elo_C'] = (df_chess_C_rated['white_rating'] + df_chess_C_rated['black_rating'])/2

#create a response variable for checkmate
df_chess_C_rated['checkmate_newcol_C'] = 0

df_chess_C_rated.loc[(df_chess_C_rated['victory_status'] == "mate"), 'checkmate_newcol_C'] = 1 

#create response variable for resign
df_chess_C_rated['resign_newcol_C'] = 0

df_chess_C_rated.loc[(df_chess_C_rated['victory_status'] == "resign"), 'resign_newcol_C'] = 1 

#create response variable for draw 
df_chess_C_rated['draw_newcol_C'] = 0

df_chess_C_rated.loc[(df_chess_C_rated['victory_status'] == "draw"), 'draw_newcol_C'] = 1 

#shorten x and y names for MODEL 1
checkmate_resp_C = df_chess_C_rated['checkmate_newcol_C']
pred_mod_1_C = df_chess_C_rated[["difference_C", "average_elo_C"]]

#logistic regression of checkmate against pred1 and pred2
logit_model_1_C = sm.Logit(checkmate_resp_C, sm.add_constant(pred_mod_1_C)).fit()
#print(logit_model_1_C.summary())

#shorten x and y names for MODEL 2
resign_resp_C = df_chess_C_rated['resign_newcol_C']

#logistic regression of resign against pred1 and pred2
logit_model_2_C = sm.Logit(resign_resp_C, sm.add_constant(pred_mod_1_C)).fit()
#print(logit_model_2_C.summary())

#shorten x and y names for MODEL 3 
draw_resp_C = df_chess_C_rated['draw_newcol_C']
pred_mod_3_C = df_chess_C_rated[["difference_C"]]

#logistic regression of draw agaist diference
logit_model_3_C = sm.Logit(draw_resp_C, sm.add_constant(pred_mod_3_C)).fit()
#print(logit_model_3_C.summary())

# PLOTS FOR LOGISTIC REGRESSION MODEL

#### MODEL 1 ######
#Label X's and Y
X_C = df_chess_C_rated["difference_C"].values.reshape(-1,1)
Y_C = df_chess_C_rated["checkmate_newcol_C"].values.reshape(-1,1)

#specif type of regression
LogR = LogisticRegression()

#overlay model line
LogR.fit(X_C ,np.ravel(Y_C.astype(int)))

#make scatter of x vs y
#plt.scatter(X_C, Y_C,alpha = 0.5, label = 'data')
#plt.scatter(X_C,LogR.predict_proba(X_C)[:,1], c = 'red', alpha = 0.5, label = 'model')
#
##Give Graph title and label axes
#plt.title('Checkmates vs Difference between elo Rating')
#plt.xlabel('Difference')
#plt.ylabel('Probability of A Checkmate')
#plt.legend(loc=1)
#plt.show()

#### MODEL 2 ######
#Label X's and Y
X_C = df_chess_C_rated["average_elo_C"].values.reshape(-1,1)
Y_C = df_chess_C_rated["resign_newcol_C"].values.reshape(-1,1)

#specif type of regression
LogR = LogisticRegression()

#overlay model line
LogR.fit(X_C ,np.ravel(Y_C.astype(int)))

##make scatter of x vs y
#plt.scatter(X_C, Y_C,alpha = 0.5, label = 'data')
#plt.scatter(X_C,LogR.predict_proba(X_C)[:,1], c = 'red', alpha = 0.5, label = 'model')
#
#Give Graph title and label axes
#plt.title('Resign vs Average elo Rating')
#plt.xlabel('Average elo Rating')
#plt.ylabel('Log Odds of A Resign')
#plt.legend(loc=1)
#plt.show()

#### MODEL 3 ######
#Label X's and Y
X_C = df_chess_C_rated["difference_C"].values.reshape(-1,1)
Y_C = df_chess_C_rated["draw_newcol_C"].values.reshape(-1,1)

#specif type of regression
LogR = LogisticRegression()

#overlay model line
LogR.fit(X_C ,np.ravel(Y_C.astype(int)))

##make scatter of x vs y
#plt.scatter(X_C, Y_C,alpha = 0.5, label = 'data')
#plt.scatter(X_C,LogR.predict_proba(X_C)[:,1], c = 'red', alpha = 0.5, label = 'model')
#
##Give Graph title and label axes
#plt.title('Draw vs Difference Between elo Ratings')
#plt.xlabel('Difference')
#plt.ylabel('Log Odds of A Draw')
#plt.legend(loc=1)
#plt.show()

#Query 3
#what percent of games ended in checkmate
df_check2 = df_chess_C_rated.query('victory_status == "mate"')
#5,146 ranked matches ended in checkmate

#Query 4
#what percent of games ended in resign
df_check3 = df_chess_C_rated.query('victory_status == "resign"')
#8,969 ranked matches ended in checkmate
#------------------------------------------------------------------------------



# Adonis portion 
# regression on increment and increment start time vs turn and vs opening play
df_chessAdonis = pd.read_csv("chess.csv", usecols=["rated", "turns", "increment_code", "opening_ply"])
df_chessAdonis = df_chessAdonis.query("rated == True")
lst_Increment = df_chessAdonis['increment_code'].tolist()

def increment_to_starting(lst):
    lst_output = []
    for i in range(0, len(lst)):
        lst_output.append(int(lst[i].split('+')[0]))
    return lst_output

def increment_to_increment(lst):
    lst_output = []
    for i in range(0, len(lst)):
        lst_output.append(int(lst[i].split('+')[1]))
    return lst_output

lst_Incr_Start = pd.Series(increment_to_starting(lst_Increment))
lst_Incr_Incr = pd.Series(increment_to_increment(lst_Increment))

df_chessAdonis["start"] = lst_Incr_Start 
df_chessAdonis["increment"] = lst_Incr_Incr 

lst_Incr_Start_l = lst_Incr_Start.tolist()
lst_Incr_Incr_l = lst_Incr_Incr.tolist()
lst_Turns = df_chessAdonis["turns"].tolist()
lst_Opening = df_chessAdonis["opening_ply"].tolist()

# Fit and summarize OLS model
modStartingToTurns = sm.OLS(lst_Incr_Start_l, sm.add_constant(lst_Turns)).fit()
modStartingToTurns.summary()
# p value is 1.02e-11
#plt.scatter(lst_Incr_Start_l, lst_Turns)
#plt.title('Turns vs Start Times')
#plt.xlabel('Time')
#plt.ylabel('Turns')
#plt.show()
#plt.close()

modIncrToTurns = sm.OLS(lst_Incr_Incr_l, sm.add_constant(lst_Turns)).fit()
modIncrToTurns.summary()
# p value is 1.96e-08
#plt.scatter(lst_Incr_Incr_l, lst_Turns)
#plt.title('Turns vs Increment Times')
#plt.xlabel('Time')
#plt.ylabel('Turns')
#plt.show()
#plt.close()

modStartingToOpening = sm.OLS(lst_Incr_Start_l, sm.add_constant(lst_Opening)).fit()
modStartingToOpening.summary()
# p value is 0.0693
#plt.scatter(lst_Incr_Incr_l, lst_Opening)
#plt.title('Opening Turns vs Start Times')
#plt.xlabel('Time')
#plt.ylabel('Turns')
#plt.show()
#plt.close()

modIncrToOpening = sm.OLS(lst_Incr_Incr_l, sm.add_constant(lst_Opening)).fit()
modIncrToOpening.summary()
# p value is 0.0227
#plt.scatter(lst_Incr_Incr_l, lst_Opening)
#plt.title('Opening Turns vs Increment Times')
#plt.xlabel('Time')
#plt.ylabel('Turns')
#plt.show()
#plt.close()





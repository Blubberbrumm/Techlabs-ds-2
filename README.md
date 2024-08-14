# Team_ds2
Git repository for techlabs data science team 2

07.07.2024:
 - Meeting at Ruby Carl Workspace
 - Setup of Github repository
 - Setup of Github project
 - Patrick (mentor), Irem and Taqi weren't here

18.07.2024:
 - Zoom meeting with Patrick
 - Irem couldn´t make it
 - Discussing our topic:
    We have cost-sensitive analysis (more focus on engineering). Less theory, more praxis).
    Questions:
     What do we do with our prediction? (Different focus on what the model is supposed to do)
     Is diagnosing Diabetis better than not? (Diagnosing is a thing, because it can create a lot of cost)
     What is more costly? Tell people they´re diabetic even if they´re not? Or tell sick people they don´t have diabetis?
     What classifiers in target? What is our target? Find out online what the cost of wrong classification are.
  - other group focusses how to make model fair, we analyze how cost effective moidel is
  - The Data:
     Zip file has 5 pickle files, very big files, regular data loading methods might be slow. If Panda is too slow, we can use polars, newer version that is faster, can use sql and other formats       
  -Questions:
    - Which method do we want to use?
    - Which metrics are we focussing on? How to modify the metrics?

08.08.2024:
  -  Questions we need answers to: Which models do we do? What do we analyze
  -  Tell Noparat that we can´t participate
  -  generalising is important, data doesn´t need to be perfectly cleaned up
  -  Trainingsscore vs Testscore: we have
  -  Explorative analysis: what is in data, what do we need, what do we want to model, what do we want to test for?
  -  Other group: men vs women
  -  Hypothesis should be written down, but let model choose, what points we use
  -  Solution for Missing data:
      -delete row
      -delete column
      - inputdata :
        ->add from median/mittelwert 
        -> use machine learning algorithm for input
        ->randomforestclassifier  (sklearn, unter scikit-learn)
   - understand linear regression and decision tree to get better understanding of more evolved algorithms

   14.08.2024:
   - we need to put source data in this Repo
   -  have another folder to merge Data 
   -  How to use Github: use different branches for each task (ex. cleanup data, get new folder called clean data). After finishing task, merge to give others access to the branch we worked on
        


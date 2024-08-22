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


  22.08.2024:
  
- we want to forecast, not analyse
- we do not have many features, we should use all of them and let model decide what it is
- group labels should be explained for different parameters
- xtrain:
	- is created but pushed immediately. 
	- make folds more or less balanced percentagewise. Have a close amount of each type (stratified fold)
-then randomforestclassifier: 
		-> we did no tuning, immediately put everything into it. We should look into tuning
		-> inspiriation: scikit-learn.org RandomizedsearcgCV, example above randomized classes.
		-> randomforestclassifier: bootstrap should always be true, hyperparameter should be decided by the model. balanced subsample might be good
			-> balanced doesn´t use random subsamples, it uses a balanced 
			-> leave randomstate completely out of hyperparameter tuning 
- confusion Matrix: 
	We now have 6 classes, but many are probably not useful
	do as patrick said: drop everything but 3 and 1. 3: no diabetes, 1 having diabnetes from eating 3


What patrick told us to do:
	-Do not drop features beforehand
	-let the model decide 1st
	-stratifieldkfold together with a RandomsearchCV
		->look if randomsearchcv needs stratifield kfold as a direct parameter
	-use hyperparamters of the randomforest
	-give it a useful range for the randomsearchcv to look choose from
	-Drop everything frokm diabetes target which is not 3 or 1
	-we predict who doesn´t have diabetes, but we don´t want that.
	

Presentation from Patrick:
	- machine that picks potatoes. We want to pick potatoes and not pick up stones
	- 100 objects, 95 stones and 5 potatoes. 
	- Can we correctly idenftity potatoes as potatoes and stones as stones?
	- If 95 stones are identified and no Potatoes are picked, we have 95% accuracy and no potatoes which is not good
	- to solve this we should not rebalance data, add more potatoes 	
	- Confusion matrix: 
		- true positive/ (true positive + false positive and false positive) should be used by us, sicne it´s precision
	 -Precision and recall used together, those are the metrics we want
	-How to balance precision and recall:
		Use F1 Score to have a balance betweem precision and recall. Then based on the imbalanced apsect
		-recall: forecast for presence of attribute
		-specificity: forecast the absence of attribute
		- precision is different from Specificity
		- Then go online and check on the internet what is more costly depending on who this is for
			
Why recall might be wrong tool:
 1. 300000 diabetes
 2. 30 no diabetes
 3. always output have diabetes
 4. 300030 entries with a 1
 5. Recall = 1

Mini-Homework:
	Before fiddling around, give patrick value of homework in the chat

Presentation requirements: roughly 5 slides, speak in pictures not text, explain to people


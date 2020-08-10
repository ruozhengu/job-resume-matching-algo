

## Job Title Prediction: Learn Knowledge For Job-Candidate Matching

**Author: Ruozhen (Gabriel Gu)**

**School of Computer Science**

**University of Waterloo**

##### Abstract:

Resume ranking and job matching nowadays are very proficient and important in the recruitment process. Traditional engines such as Applicant Tracking System has been introduced but they are limited by keyword filtering which may not be adequate across a large scale of application pool data. Other Recurrent Neural Network based natural language processing solutions are presented but they usually require large computational time to train and suffer from vanishing gradient problems. More importantly, those methods heavily rely on human manual work on text data parsing, feature engineering and domain expertise. Nevertheless, all the current solutions for job matching are targeted for employer. As a result, we are presenting a state-of-art deep learning solution targeted not only for employers, but also for all job seekers with an aim to fill them in with the ideal job vacancies and maximize their potentials during the job hunting process. We achieved this by combining Convolutional Neural Network with Learning To Rank approach to generate a job title and job description embeddings, which can be trained in relatively short periods of time and requires minimal human interventions and no domain experts. 

 

##### 1. Introduction

As of the 2020's COVID-19 situation, the importance of intelligent recruitment processes is being emphasized in many industrials. One of the problems hiring managers are dealing with is to create an optimal and ideal match for candidates and vacancies in highest measures. Traditional resume parsing algorithms have been introduced but they rely on fixed rules and those models require huge manual work and are not kept pace with rapid development of the machine learning techniques. So the questions remains that if a more automated method can be adopted in order to learn information solely rely on raw job description data with minimal feature engineering requested.

To address this problem, we have designed a Convolutional Neural Network which could effectively extract and learn feature factors out of job description embeddings. The framework of this solution is constructed by several NLP techniques including Word Embeddings and is mainly based on Keras. In the following parts of paper, we will introduce the dataset built based on Indeed Canadian Database in section 3 and also the model architect in section  4 with experimental results in section 6.

##### 2. Related Work

Job matching has been an ever-present topic since the age of online recruiting. It is basically an applicant tracking system which has already been widely used by online recruitment platforms such as Linkedin and Taleo. In most cases, the applicant tracking system collects and stores resumes in a database for recruiters to sort and search using a variety of searching criteria. This is a more conventional approach for the problem. As pointed out in a paper [2], the main challenges and limitations to this approach are the real-time computing (query) latency for scoring millions of structured candidate job documents, the difficulties of incorporating different types of user interaction signals and the balance between delivering sufficient but not too many number of applications based on job description.

As of today, many researchers have been conducted on this issue using modern machine learning and deep learning methods. Among them, Schmitt et al. [3] used a hybrid system where a deep neural network was trained to match the collaborative filtering representation properties. In a more recent research by Huynh et al. [4], different deep neural network models, TextCNN, Bi-GRU-LSTM-CNN, and Bi-GRU-CNN, were assessed and an ensemble model combining them was created. Compared to examples of other researchers‘ projects, our model has significant differences on detailed design of the neural network, word embeddings, data feature extraction and result ranking. While yielding better results than previous works, their models still fail in ranking the applicants. They are more focused on labelling and matching rather than ranking. Our model not only matches qualified applicants to post jobs but also gives a rank on each application based on both the resume’s content and the job description. Moreover, rather than a limited and narrow dataset like the IT job dataset in Huynh et al. [4], our dataset is composed from a variety of sources and job sectors which in theory will make our model more robust and generic.

##### 3. Dataset

In this paper, we have implemented a web scraper in Python to legally scrape the Indeed.ca website for 13490 recent job postings in 2020. You can find source code of scraper on Github 【1】.The search query is conducted with filters on Top 100 major cities 【2】 in Canada to get full listings of open positions shuffled by posted date. After basic data cleaning and removing duplicates, we eventually prepared 10000 distinct job descriptions with for training. A detailed features of each record is showed in Figure 1. We have only included 3 examples as a reference.

<table>
  <tr>
    <th>Title</th>
    <th>Company</th>
    <th>Job Description</th>
    <th>Label</th>
  </tr>
  <tr>
    <td>Team Support Assistant (Remote)</td>
    <td> HR Johanna Grey</td>
    <td>As a Team Support Assistant, you will perform team needs operations such as data entry. Checking/updating query list orders; providing administrative support to the team...</td>
      <td>Assistant</td>
  </tr>
  <tr>
    <td>Junior Administrator</td>
    <td>Spiritleaf</td>
    <td>the Junior Administrator is responsible for assisting Spiritleaf marketing team with a variety of administrative duties and shipping/receiving to support the marketing efforts for both consumers and Spiritleaf locations from coast-to-coast...</td>
      <td>Admin</td>
  </tr>
  <tr>
    <td>Receptionist / Service Cashier</td>
    <td>AutoCanada</td>
    <td>We are looking for a reliable receptionist who can work Monday thru Thursday, 8am to 5pm & Fridays 9am-6pm.The position is answering phones, greeting our customers, inventory maintenance and stocking in vehicles, daily cash reconciliation & posting...</td>
      <td>Cashier</td>
  </tr>


Figure 1: table schema for job data scraping

The top 3 categories of job labels are: Retail, Customer Service and Cashiers. The total number of words in the dataset are around 1 million. The dataset is split into a set for training (8.5k) and a set for testing (1.5k). Minimal another preprocessing has been conducted except of word tokenization and turning strings to lowercases. 

##### 4. Methodologies

In this study, we will propose our model architecture, which involves a convolutional neural network (CNN) to generate word embeddings for job descriptions and another lookup table for job title embeddings are also created. We match two embeddings using cosine similarity as a scoring function.

**4.1 Job Description Layers**

As calculated, we have around 1 million word matrix table and they are tokenized through Kera's Tokenizer API to be converted to word index. Sentence paddings are applied with a max sequence length limit of 250. We have read in pre-trained Glove Word Embeddings and converting each word index into corresponding embedding vectors for later training. To prepare for the CNN model, we load pre-trained word embeddings into an Embedding layer for job description with input length constraints. The final look up tale contained 100 dimensional embeddings for the most frequent 23845 words. 

As the model architect displayed in Figure 2, we have passed the word embedding layers into a convolution layer with filters equals to 1000, kernel width set to 5 and stride set to 1. Max pooling are applied to reduce parameters.  Dropout of 0.3 is used for regularization purpose. Eventually we apply dense layers with RELU activation function to pass information and get a 100 dimensional data matrix containing job description learnt information.  Up to this step, We will still need to process job title table data into a 100 dimensional matrix for ranking.

![image-20200808032747787](C:\Users\gugab\AppData\Roaming\Typora\typora-user-images\image-20200808032747787.png)

Figure 2: CNN Model Architect

**4.2 Job Title Layers**

Compared to job description embedding, it is slightly easier to convert and transform job title look up table. The title embedding matrix is achieved by getting a label encoder of previous training data on "Label" column and then pass the encoder index into the Glove Work Embedding. Again, to match job description embedding, we defined the output shape as 100 dimensional matrix. 

Along with basic model configuration, we have also defined cosine similarity comparison functions as well as some callback functions to add model checkpoints such as for early stoppings. We have turned our rest of model into a ranking problem.

**4.3 Training**

With the pre-built job title and job description embedding layers, to connect and train the model, we will turn this into a supervised learning to rank (LTR) problem. LTR is usually a common techniques applying when we need a training set that has appropriate relevance labels associated with the records [*]. In our example, the training data consists of both queries (job description) and labels (job title) and each query has a few of label descriptions associated with it (for example, a description related to data analysis can have "data scientist" or "data analyst" associated as labels). Formally, we denote a query pair Q as 

Q = {(Di, Ti), yi}^m for i = 1 to N.

Where Di denotes the ith job description for  N examples. Ti denotes a set of all possible job titles Di = {D\_{i,1}, D_{i, 2} ... D\_{i, Ni}}. and yi is a set of corresponding ranking scores. yi = {y\_{i,1}, ... y_\_{i, Ni}}

Provided a new job description and a number of job titles represented by feature vector X, our goal is to learn a ranking algorithm/function F that can assign rank scores to each of those job titles. Specifically, 

S(D, T) := F(X)

The actual training adopts the pairwise LTR approach with cross-entropy loss function. It uses the ordering of job titles and builds new feature instances as the preferred pairs of feature vectors. The final rankings are based on a scoring function using cosine similarity between each of such job title and descriptions pairs. The pair ranked as highest amongst all other preferred pairs is considered as the predicted labels.

The entire implementation is done using Keras and optimization are applied through stochastic gradient descent and we apply dropouts for regularization purposes. 

##### 5. Model Analysis

We will analyze the model performance by checking how well it is able to extract the job information for title predictions. And most importantly, we will further prove how the model fulfills the job by looking at some testing data, feature extractor in CNN

**5.1 Job Title Prediction Result**

Figure 3 is a randomly job description of project management role that we picked from company website which is not included in the training data. It will give us a great insights if we could run the prediction model on it and see what are the top closest job title to be generated:

<img src="C:\Users\gugab\AppData\Roaming\Typora\typora-user-images\image-20200808041433638.png" alt="image-20200808041433638" style="zoom:200%;" />

Figure 3: Project Manager Position Job Description

Notice that the key word "Project" and "Management" are actually appear inside the description paragraph. We will try to replace "Project Manager" to "role" and replace keyword "project" to "task". These are the 5 most likely job titles predicted by the model for the original job description as well as the one after keyword replacement. 

<table>
  <tr>
    <th>index</th>
    <th>Original</th>
    <th>Filter Out Keyword</th>
  </tr>
  <tr>
    <td>1</td>
    <td> Project Manager</td>
    <td>Product Manager</td>
  </tr>
  <tr>
    <td>2</td>
    <td>Project coordinator</td>
    <td>Project Manager</td>
  </tr>
  <tr>
    <td>3</td>
    <td>Product Manager</td>
    <td>IT Product Manager</td>
  </tr>
    <tr>
    <td>4</td>
    <td>PMO</td>
    <td>Operations Coordinator</td>
  </tr>
    <td>5</td>
    <td>Operation Manager</td>
    <td>Senior Portfolio Lead</td>
  </tr>
</table>

Figure 4: title predictions for new job description example with keyword kept and removed

As visualized above, our model has performed well for both scenarios and it is robust enough to predict job title without any dependency on existing keywords. To be exact, since that without any filtering, keyword "project" has appeared a few times and it is reasonable for the model to directly infer to "project manager" role. For the case with the filtering applied, "project" is no longer visible to the model but the sentence has a few relevant sentences to "product", which makes our model to predict "product manager" as the top score. In reality, project manager and product manager do not have a clear boundaries and this shows that our model was able to extract job title but also provide a meaningful prediction results with more direct information removed (job title keyword in this example). 

Next, we would like to test the model on a few new job descriptions that are structured differently: one sentence string description without any job title information. Figure 5 has illustrated the inputs and results.

<table>
  <tr>
    <th>Input</th>
    <th>Predicted</th>
  </tr>
  <tr>
    <td>Operators for ground works</td>
    <td>Groundworker</td>
  </tr>
  <tr>
    <td>Take care of young age child at families</td>
    <td>Nanny</td>
  </tr>
  <tr>
    <td>Completition of CFA Level 1 with basic reporting and anlytical experience</td>
    <td>Financial Analyst</td>
  </tr>
    <tr>
    <td>We look for someone with 3 years Java programming with </td>
    <td>Java Developer</td>
         <tr>
<td>We look for someone with 3 years Python programming with </td>
    <td>Python Developer</td>
    </tr>
    <td>Experience with deep learning knowledge and have basic exposure to some ML tools and libraries</td>
    <td>Data Scientist</td>
      </tr>
<td>XYZ technical operation</td>
    <td>Operator</td>
  </tr>
</table>

Figure 5: Six additional simply structured queries with predicted titles

Above example demonstrates that our model does the predictions mainly based on what a worker usually does for a specific job type. To be exact, it predict someone does Java programing to be a Java developer and learns that someone usually takes care of child as nanny. By simply replacing single word from "Java" to "Python", the model will gives two different programming roles. As for the last example, XYZ is a fake name not exist in the training data, so it makes sense for our model to predict it as an "operator".

**5.2 Job Analogies**

Recall that before the ranking, two by-products of our model is the embedding matrix for job descriptions as well as job titles. For a more intuitive visualization of the embedding's quality, we have randomly picked three titles and listed their close neighbors in the scope of embeddings.

<table>
  <tr>
    <th>Hiring Manager</th>
    <th>Data Analyst</th>
      <th>Statistical Analyst</th>
  </tr>
  <tr>
    <td>Talent Acquisition Specialist</td>
    <td>Data Analytics</td>
      <td>Statistician</td>
  </tr>
  <tr>
    <td>Hirring Manager</td>
    <td>Data Scientist</td>
      <td>Statistical analyst</td>
  </tr>
  <tr>
    <td>Recruitment Manager</td>
    <td>Data Science</td>
      <td>Research Statistician</td>
  </tr>
    <tr>
    <td>Hiring specialist</td>
    <td>Maching Learning Developer</td>
        <td>Biostatistician</td>
         <tr>
    <td>Talent Acquisition Associate</td>
    <td>Big Data Analyst</td>
    <td>Data Scientist</td>
      </tr>
</table>

Figure 6: Three Job Analogies found in the job description embedding space

As shown, we have found the job titles in various categories and analogies. While the order of close-neighbors shown above does not follow a specific order, we can find that our model is tolerant to spelling errors in some extent (for example "Hirring Manager" versus "Hiring Manager"). Our team has also found that our embeddings also follow closely to cosine distance by taking the trained word and job title embeddings from the model. For example: `cos(w("nanny") - w("baby") + w("programming"),  w("software Programmer"))`.

**5.3 Feature Detector**

Recall that two of our main goals is first to eliminate the feature engineer and also learn/extract enough information from job descriptions and saved in each neuron so that later we can perform more operations using the knowledge (for example to match the candidates with job vacancies). It is worth examining the first convolutional layer (refers as feature detector in following context) by checking how it is trained to be active upon a job pattern is observed. 

A easy way of checking this will be finding top 5 input patterns that correspond to the maximal activation in the neuron, which are shown in the Figure 7 and 8.

<table>
  <tr>
    <th>Office Manager</th>
    <th>Clean Staff</th>
  </tr>
  <tr>
    <td>answering phone events typing filling</td>
    <td>toilets ironing children landings</td>
  </tr>
  <tr>
    <td>earner he with typing schedule provide</td>
    <td>guests toilet cleaning kids taking</td>
  </tr>
  <tr>
    <td>answer handling calls earner clients</td>
    <td>duties maintenance janitor cleaning rubbish</td>
  </tr>
</table>

Figure 7: Feature Detector focusing on job duties

<table>
  <tr>
    <th>Supervisor</th>
    <th>Director</th>
  </tr>
  <tr>
    <td>Supervise staff valid homework</td>
    <td>have shown leadership teams of members</td>
  </tr>
  <tr>
    <td>motivate and supervise staff schedule</td>
    <td>mentoring members track performance</td>
  </tr>
  <tr>
    <td>oversee and manage welders junior </td>
    <td>proven time management lead floor</td>
  </tr>
</table>

Figure 8: Feature Detector focusing on skill sets

We have found that some feature detectors focus on the skill sets while others might focus on duties or other factors including language and locations. Interestingly, we also find detectors looking for certain candidate manners as well as physical behaviors.

Figure 7: Feature Detector focusing on job duties

<table>
  <tr>
    <th>Manner & Appearance</th>
  </tr>
  <tr>
    <td>Asain hair clean apperance</td>
  </tr>
  <tr>
    <td>English speaking asian accents</td>
  </tr>
  <tr>
    <td>presentable grommed clean disposition</td>
  </tr>
</table>

Figure 8: Feature Detector focusing on manners

These examples have demonstrated that our CNN filters instead of looking for a specific keyword or activation phrases, it has covered a huge variety of job features in all all descriptions. While our CNN model are taking raw, uncleaned and no feature pre-processed job information, it is proven to work excellently on information extractions and representations.

**6. Experimental Result**

Based on our above model proposal, we could use it to build a few applications including job title predictions and candidate-resume matchings. In our research, we have created such simple application to learn and predict a specific title of occupation based on job descriptions. Also, we compare the model performance with some of the other popular machine learning models on a job description dataset proposed by Papachristou, which contains ten thousand distinct job description with more than twenty-five job types. As shown in the Figure 10, it is compelling to see that our CNN approach has out-performed other models regarding its precision, accuracy and recall values.



**References**

[1]  Ganesh Venkataraman Krishnaram Kenthapadi, Benjamin Le.  Personalized job recommenda-200tion system at linkedin:  Practical challenges and lessons learned.RecSys, August 27–31(1):201346–347, 2017.

[2]  Philippe  Caillou  Thomas  Schmitt  and  Michele  Sebag.   Matching  jobs  and  resumes:  a  deep203collaborative filtering task.EPiC Series in Computing, 41(1):124–137, 2016

[3]  Ngan  Luu-Thuy  Nguyen  Anh  Gia-Tuan  Nguyen  Tin  Van  Huynh,  Kiet  Van  Nguyen. Job prediction:   From  deep  neural  network  models  to  applications.IEEE  RIVF,  2020.doi: http://dx.doi.org/10.1002/andp.19053221004

[4]  Shai Shalev-Shwartz and Shai Ben-David.Understanding Machine Learning: From Theory to Algorithms. Cambridge University Press, 2014

[5]  H. D. Block.  The perceptron:  A model for brain functioning.Reviews of Modern Physics, 34(1):123–135, 1962.

[6]  A. Novikoff.  On convergence proofs for perceptrons.  In Symposium on Mathematical Theory of Automata, pages 615–622, 1962






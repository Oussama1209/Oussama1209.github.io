## Abstract

Maslow's Hierarchy of Needs serves as a model for assessing human behavior, offering insights into the profound shifts witnessed during the pandemic. From altered relationship dynamics to revised safety measures and changing professional priorities, our needs underwent a transformative evolution. The pivotal consideration lies in whether the data collected during this critical period truly encapsulates our emotions and intuitions. Is it aligned with Maslow's enduring theory, a testament to its continued dominance in shaping our daily lives? Our study delves into the pandemic-induced changes, exploring whether Maslow's framework remains a relevant and accurate tool for understanding these shifts. Our research only encompasses EU countries. By examining the intricacies of the data at hand, we aim to uncover the resonance between our evolving needs and the timeless principles outlined by Maslow's Hierarchy, providing a nuanced perspective on the enduring impact of his theory. 

## Research Questions

1. Is Maslow’s “Hierarchy of Needs” theory still applicable during times of abnormalcy?
2. Can a pandemic cause a shift in human needs?
3. Is the base of Maslow’s pyramid (Physiological Needs) completely stable?
4. In what ways might the pursuit of self-actualization be affected by external disruptions, and how can individuals adapt their goals during challenging times?
5. How did the pandemic affect our relationships and our love and belonging needs?

## Additional Datasets

1. **EU Employment Dataset:**
   - Job applications and searches fall outside the scope of Wikipedia and are integral to Maslow’s 'safety' category. The selected dataset is from Kaggle, specifically the Unemployment in European Union data from the EU open data portal. The analysis focuses on EU citizens due to data availability, with extensive information from Wikipedia searches in European languages. The study utilizes data from 1983 to 2021, specifically during the Covid-19 pandemic, using prior years to establish a baseline for comparison, considering seasonal effects and volume variations (difference in differences).

2. **Google Trends:** 
3. **Wikipedia Pageviews**
   - Given the broad categorization of topics, we will incorporate both Wikipedia pageviews and Google Trends data to assess the relevance of more specific and niche subjects. The data will span from 2019 to 2021, allowing for an examination of trend evolution before and during the pandemic.
   - Examples of searches relevant to assessing the evolution of each level of Maslow’s pyramid include:
      - **Self-Actualization:** mindfulness, time management, steps to reach life goals,...
      - **Esteem:** communication skills, building a respectful environment,...
      - **Love and Belonging:** online dating, long-distance relationships, divorce,...

## Methods

<p> Recognising changes in Maslow’s hierarchy of needs requires for us to underline fluctuations in the interests of people’s searches (and therefore needs). Capturing such changes requires a rigorous methodology, we will propose a methodology that will provide a holistic view of relative change in needs through a difference in differences approach. This approach enables us to quantify differences that are due to the pandemic and its related restrictions on society. </p>


- **Data Cleaning:**
   - Since our research focuses exclusively on EU countries, it is essential to eliminate all non-EU countries from our datasets.
   - The time stamps are converted to date-time for easier visualization and analysis.
   
- **Initial Analyses:**<br>
Regarding the 'aggregated timeseries' dataset, we opted to focus on a specific country with robust data: Italy. Initially, we illustrated the overall monthly evolution of Wikipedia page visits from 2019 to 2020. Upon observing unusually high activity during the COVID-19 period, we delved deeper into specific categories, finding a similar trend.

To quantify this evolution, we undertook two approaches: first, conducting linear regression on our plots to understand the impact of lockdown on view numbers, and second, performing hypothesis testing.

Both procedures indicated that, on a per-topic basis, as well as for general page visits, there was an increase during the COVID-19 period.

- **Data Categorization:**
   - To align Wikipedia topics with the corresponding levels on Maslow's Hierarchy of Needs, we will implement a systematic categorization process. Wikipedia topics and Google Trends will predominantly inform these categories, while Mobility data will contribute to understanding Physiological Needs, and EU Unemployment data will be instrumental in assessing Safety Needs.

- **Difference-in-Differences Analysis:**
   - Our examination will explore the changing levels of interest in the categorized topics, correlating with significant timestamps (lockdown, first death, etc). These patterns will be linked to corresponding human needs [1].
   
## Proposed Timeline

An exploratory data analysis performed on the data sets that were chosen, mostly completed in milestone 2.

**Step 1: Categorize Wikipedia Topics**

Associate each Wikipedia topic with one of Maslow's Pyramid levels, primarily focusing on Self-Actualization, Esteem, and Love and Belonging.

**Step 2: Select Appropriate Topics from Google Trends and Wikipedia Pageviews for Categorization**

**Step 3: Establish the Connection between Mobility Trends and the Base of the Pyramid (Physiological Needs)**

Examine mobility data related to grocery stores/pharmacies to gain insights into the evolution of Physiological needs during the pandemic. Through a comparative analysis of various mobility data sets, conclusions can be drawn regarding the stability of the base of Maslow's pyramid.

**Step 4: Establish the Connection between EU Employment Rates and Safety Needs**

Safety needs, revolving around security and protection, encompass aspects like employment status. [2] Analyzing EU employment rates during and post-COVID allows us to explore the development of safety needs.

**Step 5: Perform a Difference in Differences Analysis**

Utilize a difference in differences analysis and other time-dependent methodologies to quantify changes in people's needs before and after the pandemic.

**Step 6: Determine Whether the Pyramid Has Shifted**

Conduct a comprehensive analysis to redefine the hierarchy of needs, normalizing changes in each category for meaningful comparisons.


![Cat](pyramid-data.png)

## Organisation
Our repository will consist of two ipynb files. The helpers which consists of functions that can be applied to all datasets to format them in a usable shape and form. The Project main file which consists for now of the initial exploratory data analysis that is performed in the scope of milestone 2.

## Work Repartition
Aly: Preliminary Data Analysis of of Apple Mobility Trends Dataset and of Web-Scraped Pageview Datasets. Multiple European Countries.

Ellen: General Scope of project, Global mobility report analysis, Google Trends analysis for Physiological needs

Kamal: Preliminary Data Analysis (Italy case study). EU Unemployment Analysis.

Nico: Preliminary EU unemployment Analysis, Difference in Differences for apple mobility, EU Unemployment and wikipedia pageviews concerning Self-actualization, Love and Belonging and Esteem.

Oussama: Web-scrapping of pageview count in different countries of the top three levels in Maslow's hierarchy.
## References

[1] Jina Suh, Eric Horvitz, Ryen W. White, Tim Althof (2021). Population-Scale Study of Human Needs During the COVID-19 Pandemic: Analysis and Implications <br>

[2] Maslow’s safety needs: Examples &amp; definition - study.com. Available at: https://study.com/academy/lesson/maslows-safety-needs-examples-definition-quiz.html

# Data story Link
https://kamalnour.github.io/adatastory/

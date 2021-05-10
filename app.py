import pandas as pd
import streamlit as st
import streamlit.components.v1 as stc

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel



RESULT_TEMP = """
<style>

</style>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/css/bootstrap.min.css" integrity="sha384-B0vP5xmATw1+K9KRQjQERJvTumQW0nPEzvF6L/Z6nronJ3oUOFUFpCjEUQouq2+l" crossorigin="anonymous">

<div class="row" style="height:100%; font-family:'Arial'!important; border-radius: 20px;">
    <div class="col-5" style="height:100%">
        <div class="card" style="height:100%; border:none">
            <div class="card-body">
                <h5 class="card-title" style="font-weight:800">{}</h5>
                <p><span class="badge bg-danger text-white">{}</span></p>
                <p class="card-text" style="font-style: italic;">{}</p>
                <span class="badge bg-warning">{}</span>
            </div>
        </div>
    </div>

    <div class="col-4" style="height:100%">
        <div class="card" style="height:100%; border:none; font-size: 12px;">
            <div class="card-body">
                <h5 class="card-title" style="font-weight:800">Composition</h5>
                <p>Energie /100g: <strong>{} Kj</strong></p>
                <p>Graisse /100g: <strong>{} g</strong></p>
                <p>Sucre /100g: <strong>{} g</strong></p>
                <p>Fibre /100g: <strong>{} g</strong></p>
                <p>Proteine /100g: <strong>{} g</strong></p>
                <p>Sel /100g: <strong>{} g</strong></p>
                <p>Nutrition score: <strong>{}</strong>
            </div>
        </div>
    </div>
    <div class="col-2" style="height:350px; display: flex; justify-content: center; align-items: center; text-transform:uppercase">
        <span class="badge bg-primary text-white text-center" style="font-size:30px">{}</span>
    </div>
</div>


"""



# Functions
# Vectorize + cosine similarities Matrix
# Recommendation System
# Search for a product
def load_data(data):
    df = pd.read_csv(data)
    df = df.drop('Unnamed: 0', axis=1)
    return df

@st.cache
def search_term_if_not_found(term, df):
    result_df = df[df['content'].str.contains(term)]
    return result_df


def vectorize_text_to_cosine_mat(data):
    cv = CountVectorizer()
    cv_matrix = cv.fit_transform(data)
    # Get the cosine
    cosine_sim = cosine_similarity(cv_matrix)
    return cosine_sim

@st.cache
def get_recommendation(content, cosine_sim_mat, df, num_of_rec=5):
    #indices of the product
    product_indices = pd.Series(df.index, index=df['product_name']).drop_duplicates()
    # Index of the product
    idx = product_indices[content]
    # Look into the cosine matrix for that index
    sim_scores = list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    selected_product_indices = [i[0] for i in sim_scores[1:]]
    selected_product_scores = [i[1] for i in sim_scores[1:]]

    result_df = df.iloc[selected_product_indices]
    final_df = result_df[
        [
            'product_name', 
            'brands', 
            'categories_en', 
            'ingredients_text', 
            'allergens', 
            'nutrition_grade_fr', 
            'energy_100g',
            'fat_100g',
            'sugars_100g',
            'fiber_100g',
            'proteins_100g',
            'salt_100g',
            'nutrition_score_fr_100g',
            'nutrition_grade_fr'
        ]
    ]
    return final_df[:num_of_rec]

def main ():
    st.title("Foodflix Recommendation App")

    menu = ['Home', 'Recommend', 'About']
    vectorizer = ['CountVectorizer','Tfidf', 'Word Embedding']

    choice = st.sidebar.selectbox("Menu", menu)

    df = load_data('foodflix.csv')

    if choice == "Home":
        st.subheader("Home")
        st.dataframe(df.head(10))

    elif choice == "Recommend":
        st.subheader("Recommend Product")
        cosine_sim_mat = vectorize_text_to_cosine_mat(df['content'])
        search_term = st.text_input("Search for a product")
        num_of_rec = st.sidebar.select_slider('Slide to selection number of product', options=[3,4,5])
        vectorizer_choice = st.sidebar.radio('Vectorizer', vectorizer)
        if st.button("Recommend"):
            if search_term is not None:
                try:
                    results = get_recommendation(search_term, cosine_sim_mat, df, num_of_rec)
                    with st.beta_expander("Results as JSON"):
                        results_json = results.to_dict('index')
                        st.write(results_json)
                    for row in results.iterrows():
                        rec_product_name = row[1][0]
                        rec_product_brands = row[1][1]
                        rec_product_category = row[1][2]
                        rec_product_descrip = row[1][3][0:165] + "..."
                        rec_product_energy = row[1][6]
                        rec_product_fat = row[1][7]
                        rec_product_sugar = row[1][8]
                        rec_product_fiber = row[1][9]
                        rec_product_protein = row[1][10]
                        rec_product_salt = row[1][11]
                        rec_product_nutriscore = row[1][12]
                        rec_product_nutrigrade = row[1][13]

                        stc.html(RESULT_TEMP.format(rec_product_name,rec_product_brands,rec_product_descrip,rec_product_category,rec_product_energy,rec_product_fat,rec_product_sugar,rec_product_fiber,rec_product_protein,rec_product_salt, rec_product_nutriscore, rec_product_nutrigrade), height=350)
                except:
                    results = "Not found"
                    st.warning(results)
                    st.info("Suggested Options include")
                    result_df = search_term_if_not_found(search_term, df)
                    st.dataframe(result_df)    
    
    else:
        st.subheader("About")
        st.text("Built with Streamlit & Pandas")






if __name__ == '__main__':
    main()
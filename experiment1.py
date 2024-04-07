import pandas as pd
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

csv_path = r"C:\Users\prath\Downloads\CODING\Healthsync\Conflict\drugs_for_common_treatments.csv"
drugs_data = pd.read_csv(csv_path)

user_data_path = r"C:\Users\prath\Downloads\CODING\Healthsync\Conflict\User.csv"
user_data = pd.read_csv(user_data_path)

# Load substitute data
substitute_data_path = r"C:\Users\prath\Downloads\CODING\Healthsync\Conflict\Substitue.csv"
substitute_data = pd.read_csv(substitute_data_path)

# Load side effects data
side_effects_data_path = r"C:\Users\prath\Downloads\CODING\Healthsync\Conflict\drugs_side_effects_drugs_com.csv"
side_effects_data = pd.read_csv(side_effects_data_path)
def get_top_drugs(disease, drugs_data):
    # Drop NaN values from 'medical_condition_description' column
    drugs_data_cleaned = drugs_data.dropna(subset=['medical_condition_description'])

    vectorizer = TfidfVectorizer()
    all_text = list(drugs_data_cleaned['medical_condition_description']) + [disease]
    vectors = vectorizer.fit_transform(all_text)

    similarity_scores = cosine_similarity(vectors[-1], vectors[:-1])

    most_similar_index = similarity_scores.argmax()

    relevant_disease = drugs_data_cleaned.iloc[most_similar_index]

    relevant_drugs = drugs_data_cleaned[drugs_data_cleaned['medical_condition'] == relevant_disease['medical_condition']]
    top_drugs = relevant_drugs.sort_values(by='rating', ascending=False).head(3)

    response = "\nTop Drugs:\n"
    for _, drug_info in top_drugs.iterrows():
        response += f"Drug: {drug_info['drug_name']}, Rating: {drug_info['rating']}\n"

    return response, top_drugs


def get_user_record(username, password, user_data):
    return user_data[(user_data['username'] == username) & (user_data['password'] == password)]

def update_user_record(username, password, diagnosis, user_data):
    user_index = get_user_record(username, password, user_data).index[0]

    # Check for available slots (old, new_1, new_2)
    available_slots = ['old_1', 'old_2', 'old_3', 'new_1', 'new_2']
    for slot in available_slots:
        if pd.isnull(user_data.loc[user_index, slot]):
            user_data.at[user_index, slot] = diagnosis
            user_data.to_csv(user_data_path, index=False)
            return slot

    return None  # All slots are full

def extract_drugs_from_string(drugs_string):
    # Assuming the string is in the format: Drug: [drug_name], Rating: [rating]\n
    drugs_list = drugs_string.split('\n')[1:-1]
    drugs_info = [info.split(', ') for info in drugs_list]
    drugs_df = pd.DataFrame(drugs_info, columns=['Drug', 'Rating'])
    return drugs_df


def substitute_top_drugs(matching_drugs, substitute_data):
    substituted_drugs = {}
    drugs_df = extract_drugs_from_string(matching_drugs)
    new_drug_info = pd.DataFrame({'Drug': ['doxycycline'], 'Rating': [9.0]})  # Add your desired rating
    drugs_df = pd.concat([drugs_df, new_drug_info], ignore_index=True)
    drugs_df = drugs_df.drop(0)
    print("drugs info")
    print(drugs_df)
 

    for _, drug_info in drugs_df.iterrows():
        drug_name = drug_info['Drug']
        print("here")
        print(drug_name)
        print("next")
        print(side_effects_data['drug_name'])
        print(side_effects_data.columns)
        side_effect_info = side_effects_data[side_effects_data['drug_name'] == drug_name]

        if not side_effect_info.empty:
            side_effects = side_effect_info.iloc[0]['side_effects']
            substituted_drugs[drug_name] = side_effects
            print(substitute_data)

            # Get substitute drug from substitute_data
            substitute_drug_info = substitute_data[ substitute_data['name'] == drug_name ]
            if  not substitute_drug_info.empty:
                substitute_drug_name = substitute_drug_info.iloc[0]['substitute']
            
               
                print(substitute_drug_name)
                substituted_drugs[substitute_drug_name] = side_effects

                return substitute_drug_name

def main():
    global user_data
    all_slots_filled = True

    st.title("Disease Diagnosis App")

    st.write("Are you a new or existing user?")
    user_type = st.radio("Select user type:", ["New", "Existing"])

    if user_type == "New":
        # Get user information
        username = st.text_input("Enter a username:")
        password = st.text_input("Enter a password:", type="password")

        # Create a new user record
        new_user = pd.DataFrame({'username': [username],
                                 'password': [password],
                                 'side_effect1': [""],
                                 'side_effect2': [""],
                                 'side_effect3': [""],
                                 'side_effect4': [""],
                                 'side_effect5': [""],
                                 'old_1': [""],
                                 'old_2': [""],
                                 'old_3': [""],
                                 'new_1': [""],
                                 'new_2': [""]})

        # Append the new user record to the user_data DataFrame
        user_data = pd.concat([user_data, new_user], ignore_index=True)
        user_data.to_csv(user_data_path, index=False)

        st.write("New user created successfully. Please log in with your credentials.")
    elif user_type == "Existing":
        # Get user login information
        username = st.text_input("Enter your username:")
        password = st.text_input("Enter your password:", type="password")

        # Check if the user exists
        user_record = get_user_record(username, password, user_data)

        if not user_record.empty:
            user_index = user_record.index[0]

            # Check for available slots
            available_slot = None
            if all_slots_filled:
                available_slot = update_user_record(username, password, None, user_data)
            else:
                for i in range(1, 4):
                    old_col = f'old_{i}'
                    if pd.isnull(user_data.loc[user_index, old_col]):
                        available_slot = old_col
                        break
                else:
                    all_slots_filled = True

            if available_slot:
                user_disease = st.text_input("Enter your Disease:")
                if user_disease:
                    result_drugs, top_drugs = get_top_drugs(user_disease, drugs_data)

                    # Update the user record with symptoms, diagnosis, and side effects
                    st.write(f"Top Drugs:\n{result_drugs}")
                    update_user_record(username, password, user_disease, user_data)

                    # Check for matches with existing diseases
                    existing_diseases = [user_data.loc[user_index, f'old_{i}'] for i in range(1, 4) if not pd.isnull(user_data.loc[user_index, f'old_{i}'])]

                    for existing_disease in existing_diseases:
                        matching_drugs, _ = get_top_drugs(existing_disease, drugs_data)
                        print(matching_drugs)
                       
                        substituted_drugs = substitute_top_drugs(matching_drugs, substitute_data)

                        
                        st.text(f"Substituted Drugs for {substituted_drugs} ")
                else:
                    st.write("Please enter a valid Disease.")
            else:
                st.write("All slots are full. Please use a different account or remove one previous disease.")
        else:
            st.write("Invalid username or password. Please try again.")

if __name__ == "__main__":
    main()

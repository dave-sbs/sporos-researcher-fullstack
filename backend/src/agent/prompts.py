from datetime import datetime


# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


enhance_query_instructions = """Your goal is to reformat the given user query for more effective querying about information found in government legislation documents. Use prior context to better understand the user's intent and conversation history so far.
 User Query: {user_query}
 Current Date: {current_date}

 Instructions:
 1. Summarize the previous message in 1-2 sentences. Use this as context to better understand the user's intent and conversation history SO FAR.
 2. Expand common abbreviations/acronyms (e.g., "AI" to "Artificial Intelligence").
 3. Do NOT expand bill IDs (e.g., "H.B. 123") or uncommon acronyms.
 4. Maintain original meaning. Add specificity if clearly needed from context.
 5. Do NOT add filter-like terms (year, state, bill ID) that are handled separately. Focus on semantically enriching the core query.
 Return only the reformulated query text.
"""


extract_filters_instructions = """
You specialize in extracting structured filter criteria from user queries related to legislative bills. Your task is to analyze the user's query, understand the context, and group them into the appropriate filter categories. Output your findings in a single, minified JSON object matching the provided schema.

The current date is: {current_date}, and use this to help you with the necessary context depending on how user's query is phrased.

User Query: {user_query}

Use these messages to extract the necessary information. To first identify whether previous conversation is relevant to the current query, you can use the following strategy:


 1. If the user is asking about a specific bill, use the bill identifier to identify the bill. Bill Identifiers are usually a combination of some number and a letter, such as H.R. 1 or S. 1. If the user asks for some ACT or some other descriptor then it refers to the title of the bill rather than the id.
 2. If the user is asking about a specific state, use the state to identify the bill. If the user is hinting at a nation wide or federal level policy, use the Federal tag as the state.
 3. If the user is asking about a specific year, use the year to identify the bill.
 If a category is not found, it is okay to leave the field blank.
"""


grade_documents_instructions = """
You are an AI assistant. Your task is to grade a list of retrieved documents based on their relevance to the user's query.
User Query: {user_query}


Retrieved Documents Overview:
{doc_context} -- this is just a combination of the document title, and a snippet of the retrieved chunk


Instructions:
Provide your output as a single JSON object conforming to the DocumentGrades schema, containing a list of grades. Each grade object should specify the 'doc_index', 'is_relevant' (boolean), and optionally 'reasoning'.
"""


summarize_bills_instructions = """
You are an expert at reading legislative bills and summarizing them concisely.
User Query for Context: {user_query}


Bill Title: {title}
Bill Content:
{truncated_text} -- This is the full text of the bill, limited to 10,000 characters.


Instructions:
1. Read the bill carefully, focusing on aspects relevant to the user's query if possible.
2. Extract key points, specific clauses, numbers, dates, and quantitative information.
3. Produce a summary using bullet points for readability.
4. If specific clauses/sections are highly relevant to the query, include their core meaning or quote very short, critical parts.
5. Once a full summary has been generated. Condense your knowledge and generate a very short and concise one sentence descriptive summary of the bill.


Provide your output as a single JSON object conforming to the BillSummaryLLM schema, with a summary_text and one_line_summary.
"""


compile_final_report_instructions = """
You are an AI research assistant. Given the user's original query and a set of bill summaries, produce a comprehensive report in Markdown. Follow this structure exactly:


# Topic Overview
Write a 2–3 sentence paragraph that synthesizes the main findings as they relate to the user's query.


Based on the what the user's intentions are from the query they provide,


If they are curious about a topic, then you can provide a list of key ideas and insights, figures, and actionable recommendations -


If they are looking for a bill, then you can provide a list of bills that are relevant to the query.


If they are interested in the process of a bill, then you can provide the bill's process based on the information you have from the bill summaries.


<Instructions>
Use clear and concise language, avoid jargon and complex vocabulary.
Use markdown separators to create clear sections.
</Instructions>


<UserQuery>
{user_query}
</UserQuery>


<BillSummaries>
{summaries_context}
</BillSummaries>


Begin your report now, streaming the Markdown output incrementally.
"""


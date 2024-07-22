import google.generativeai as genai
from PIL import Image
import io
import os
import gradio as gr


genai.configure(api_key=os.environ["GOOGLE_API_KEY"]) 

# Converting input images to bytes
def input_image_setup(images):
    if not images:
        raise ValueError("No images provided")

    image_parts = []
    for image in images:
        # Open the image file
        img = Image.open(image.name)

        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        image_parts.append({
            "mime_type": "image/jpeg",
            "data": img_byte_arr
        })

    return image_parts


def improve_prompt(image_prompts, question):
    # Set up the model
    generation_config = {
        "temperature": 0,
        "top_p":1,
        "top_k":32,
        "max_output_tokens": 4096,
    }

    model = genai.GenerativeModel(model_name="gemini-1.5-pro", generation_config=generation_config)

    input_prompt = """You are an expert in writing prompts using prompting techniques."""

    question_prompt = f"""Given a chart and an accompanying question, enhance the question using advanced prompting techniques such as chain-of-thought 
    reasoning or self-consistency to provide a more comprehensive and accurate response. Divide problem into small steps. Do not include any other information in the output.

    Question : {question}"""

    prompt_parts = [input_prompt] + image_prompts + [question_prompt]
    response = model.generate_content(prompt_parts)
    return str(response.text)


def get_image_info(image_prompts, question, task_type):

    # Set up the model
    generation_config = {
        "temperature": 0,
        "max_output_tokens": 4096,
    }

    model = genai.GenerativeModel(model_name="gemini-1.5-pro", generation_config=generation_config)

    input_prompt = """You are an expert in reading and analyzing the charts."""

    question_prompt = f"""Question : {question}"""

    if task_type == "Summarization":
        question_prompt = f"""Question : {question}

                              Address the following points, you don't need to include every point or print the output in the given format, these points 
                              are just for understanding the chart: 

                              Chart Type and Structure:

                              Identify the type of chart (e.g., bar chart, line graph, pie chart).
                              Note the axes labels, units, and any legend information.
                              Mention the time period or categories represented, if applicable.

                              Main Topic and Purpose:

                              Determine the primary subject matter of the chart.
                              Infer the intended purpose or main message of the visualization.

                              Key Data Points:

                              Identify the highest and lowest values.
                              Highlight any significant outliers or anomalies.
                              Note any critical threshold values or benchmarks.

                              Trends and Patterns:

                              Describe the overall trend (e.g., increasing, decreasing, stable).
                              Identify any cyclical patterns or seasonality.
                              Mention any notable sub-trends within specific segments.

                              Comparisons and Relationships:

                              Compare different categories or time periods.
                              Identify any correlations between variables.
                              Highlight proportions or distributions, if relevant.

                              Context and Implications:

                              Consider any provided context or background information.
                              Infer potential causes for observed trends or anomalies.
                              Suggest possible implications or consequences of the data.

                              Data Quality and Limitations:

                              Note any apparent gaps or inconsistencies in the data.
                              Mention any potential biases or limitations in the presentation."""
    elif task_type == "Question Answering":
        question_prompt = improve_prompt(image_prompts,question)
      
    elif task_type == "Comparison":
        question_prompt = f"""Question : {question}

                              Address the following points, you don't need to include every point or print the output in the given format, these points 
                              are just for understanding the chart:

                              Basic Information:

                              Identify the type of each chart (e.g., bar, line, pie)
                              Note the subject matter, time periods, or categories represented in each
                              Describe the key variables or metrics being measured

                              Structural Similarities and Differences:

                              Compare the scales, units, and axes used
                              Identify any differences in data granularity or time frames
                              Note variations in how data is grouped or categorized

                              Data Trends:

                              Describe the overall trend in each chart (e.g., increasing, decreasing, fluctuating)
                              Compare the magnitude and direction of trends across charts
                              Identify any common patterns or divergences

                              Key Data Points:

                              Compare the highest and lowest values across charts
                              Identify notable outliers or anomalies in each chart
                              Highlight any significant threshold values or benchmarks

                              Relative Performance:

                              Compare performance metrics across charts (e.g., growth rates, market share)
                              Identify which chart shows better performance in relevant metrics
                              Note any crossover points where relative performance changes

                              Time-based Analysis (if applicable):

                              Compare data at specific time points across charts
                              Identify any lag or lead relationships between trends
                              Note differences in seasonality or cyclical patterns

                              Composition and Distribution:

                              Compare the distribution of data across categories
                              Identify differences in the composition of totals or wholes
                              Note any shifts in proportions or ratios between charts

                              Correlations and Relationships:

                              Identify any correlations between variables across charts
                              Compare the strength and direction of relationships
                              Note any unexpected or contradictory relationships

                              Context and External Factors:

                              Consider how external factors might explain differences
                              Identify any relevant events or conditions that could impact the comparison
                              Note any limitations in comparing the data sets"""

    prompt_parts = [input_prompt] + image_prompts + [question_prompt]
    response = model.generate_content(prompt_parts)
    return str(response.text)


def identify_task_type(image_prompts, question):

    # Set up the model
    generation_config = {
        "temperature": 0,
        "max_output_tokens": 4096,
    }

    model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)

    input_prompt = """You are an expert in reading and analyzing the charts."""

    question_prompt = f"""Given a chart and a question you have tell the question belongs to which category. Only return the category type of the question nothing else.
                          Question : {question}

                          Categories: 
                          1.  If question is related to question answering or numerical question answering based on chart return 'Question Answering'
                          2.  If question related to Chart Summarization or Chart Analysis is given return 'Summarization'
                          3.  Find number of images given, If more than 1 images are given return 'Comparison'
                      """

    prompt_parts = [input_prompt] + image_prompts + [question_prompt]
    response = model.generate_content(prompt_parts)
    return str(response.text)


def final_setup(images,question):
    image_prompts = input_image_setup(images)
    task_type_output = identify_task_type(image_prompts,question)
    task_type_output = " ".join(task_type_output.split())
    image_output = get_image_info(image_prompts, question, task_type_output)
    return image_output


def setup_gradio_interface():
    return gr.Interface(
        fn = lambda images, question : final_setup(images, question),
        inputs=[
            gr.components.File(file_count="multiple", label="Chart Images"),
            gr.Textbox(lines=2, placeholder="Enter your Question here", label="Question"),
        ],
        outputs="text",
        title="Chart Analyzer",
        description = "Perform Charts specific tasks like Summarization, Analysis, Question Answering, comparison between multiple charts and more"
    )


iface = setup_gradio_interface()
iface.launch()
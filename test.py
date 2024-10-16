import json
import streamlit as st
from openai import OpenAI

# 初始化OpenAI客户端
client = OpenAI(
    api_key=st.secrets["openai"]["api_key"],
    base_url="https://api.deepseek.com",
)

# 系统提示
system_prompt = """
你是一个专门处理股东股权变更信息的AI助手。用户将提供一段描述股东股权变更的文本。请解析文本中的"变更前"和"变更后"的股东信息,并以JSON格式输出。

对于每个股东,请提取以下信息:
- 股东名称
- 出资金额(单位:万元)
- 出资比例(百分比)

同时,请提取公司注册资本的变化信息(如果有)。

请按照以下JSON格式输出:

{
    "变更前": [
        {
            "股东名称": "股东A",
            "出资金额": 1000,
            "出资比例": 25.0
        },
        ...
    ],
    "变更后": [
        {
            "股东名称": "股东B",
            "出资金额": 2000,
            "出资比例": 40.0
        },
        ...
    ],
    "注册资本变更": {
        "变更前": 10000,
        "变更后": 12500
    }
}
"""

# 预设的例子
examples = {
    "简单例子": """
    原股东情况:张三出资100万元,占比100%。公司注册资本为100万元。
    
    本次变更:李四新增投资100万元。变更后公司注册资本增加至200万元。
    """,
    
    "中等复杂度例子": """
    原股东情况:张三出资600万元,占比60%;李四出资400万元,占比40%。公司注册资本为1000万元。
    
    本次变更:
    1. 张三减资100万元;
    2. 李四增资200万元;
    3. 新股东王五投资300万元。
    变更后公司注册资本增加至1400万元。
    """,
    
    "复杂例子": """
    原股东情况:深圳科技创新有限责任公司(以下简称"深科创")出资5000万元人民币,占比33.33%;北京未来投资有限合伙企业(有限合伙)(以下简称"北京未来")出资3000万元人民币,占比20%;张三(自然人)出资2250万元人民币,占比15%;李四(自然人)出资1500万元人民币,占比10%;王五(自然人)出资750万元人民币,占比5%;其他25名小股东共计出资2500万元人民币,占比16.67%。公司注册资本共计15000万元人民币。

    本次变更情况:
    1. 深科创增资1000万元人民币,总出资额增至6000万元;
    2. 北京未来减资500万元人民币,总出资额减至2500万元;
    3. 张三保持原有出资不变;
    4. 李四完全退出股东行列;
    5. 王五增资250万元人民币,总出资额增至1000万元;
    6. 原小股东中的10名退出,其余15名小股东增资共计500万元人民币,总出资额增至2000万元;
    7. 新引入战略投资者上海创新投资管理有限公司(以下简称"上海创投"),出资3000万元人民币;
    8. 新增自然人股东赵六,出资1500万元人民币。

    本次变更后,公司注册资本增加至18250万元人民币。各股东出资比例相应调整。
    """
}

# Streamlit应用标题
st.title("股东股权变更信息处理")

# 选择预设例子或自定义输入
input_option = st.selectbox(
    "选择输入方式:",
    ["自定义输入"] + list(examples.keys())
)

if input_option == "自定义输入":
    user_prompt = st.text_area("请输入股东股权变更信息:", height=300)
else:
    user_prompt = st.text_area("股东股权变更信息:", value=examples[input_option], height=300)

# 处理按钮
if st.button("处理信息"):
    if user_prompt:
        # 准备消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # 调用API
        with st.spinner("正在处理..."):
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                response_format={
                    'type': 'json_object'
                }
            )

        # 解析响应
        result = json.loads(response.choices[0].message.content)

        # 显示结果
        st.subheader("处理结果")
        st.json(result)
    else:
        st.error("请输入股东股权变更信息")

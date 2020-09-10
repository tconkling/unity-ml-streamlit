# Unity ML-Agents + Streamlit

Training:
- Open the Unity project, and load "RollerAgentTrainingScene"
- Run the Streamlit app: `$ streamlit run app.py`
- Within the Streamlit app: press the "Train" button
- Within Unity: Press Play
- Press Stop in Unity to end training
- The training results will be saved to `results/RollerBall/RollerBall.nn`
- To use the results in Inference, copy the results file to `Project/Assets/RollerAgentTutorial/RollerBall.nn`, replacing the existing file.

Inference:
- *Don't* run Streamlit
- Just open the Unity scene and press Play.

## Notes

- I always struggle with creating nice charts! Need to get more fluent at `pd.DataFrame`.
- Not auto-rerunning when I change non-main modules. (Modifing `mlas` modules doesn't cause a rerun.)
- Want to have just a single instance running. (I really want `@st.singleton`!)

## TODO

- A UI that actually explains what to do
- Streamlit currently hangs when trying to stop, after training
- Singleton state

## Resources

- [Unity ML Agents Github](https://github.com/Unity-Technologies/ml-agents)
- [Unity ML Agents Intro Video](https://www.youtube.com/watch?v=i0Vt7l3XrIU)
- [Unity ML Agents Getting Started Guide](https://github.com/Unity-Technologies/ml-agents/blob/release_4_docs/docs/Getting-Started.md)
- [Making a New Environment](https://github.com/Unity-Technologies/ml-agents/blob/release_4_docs/docs/Learning-Environment-Create-New.md)
- [Create your Own AI Walkthrough Video](https://www.youtube.com/watch?v=2Js4KiDwiyU)
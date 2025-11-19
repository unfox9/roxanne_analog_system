using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class WolfAgent : Agent
{
    [System.Serializable]
    public class JointControl
    {
        public ArticulationBody body;
        public float minTarget;
        public float maxTarget;
    }

    public ArticulationBody hips;
    public JointControl[] joints;

    public int decisionInterval = 5;
    int stepCount = 0;

    void FixedUpdate()
    {
        // 自己控制何時要 decision，不用 DecisionRequester
        stepCount++;
        if (stepCount % decisionInterval == 0)
        {
            RequestDecision();
        }
    }

    public override void OnEpisodeBegin()
    {
        // reset 姿勢 & 速度，簡化版先清速度
        hips.velocity = Vector3.zero;
        hips.angularVelocity = Vector3.zero;
        // TODO: 你可以把所有 joints 的 drive.target 設回中立
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // 例子：你可以照自己需要增減
        // 1. hips 高度
        sensor.AddObservation(hips.transform.position.y);

        // 2. hips 速度（local）
        var vel = hips.velocity;
        sensor.AddObservation(hips.transform.InverseTransformDirection(vel));

        // 3. 身體朝向
        sensor.AddObservation(hips.transform.up);
        sensor.AddObservation(hips.transform.forward);

        // 4. 關節狀態（每個關節一個角度）
        foreach (var jc in joints)
        {
            var drive = jc.body.xDrive;
            // 用 target 近似目前角度
            float normAngle = Mathf.InverseLerp(jc.minTarget, jc.maxTarget, drive.target);
            normAngle = normAngle * 2f - 1f; // 映射到 [-1,1]
            sensor.AddObservation(normAngle);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        var a = actions.ContinuousActions;

        // 把 action 映射成各關節的目標角度
        for (int i = 0; i < joints.Length; i++)
        {
            float x = Mathf.Clamp(a[i], -1f, 1f);
            var jc = joints[i];
            var drive = jc.body.xDrive;
            float target = Mathf.Lerp(jc.minTarget, jc.maxTarget, (x + 1f) * 0.5f);
            drive.target = target;
            jc.body.xDrive = drive;
        }

        // Reward & 結束條件（給 Python 用）
        float reward = 0f;

        // 站得越直越好
        float upright = Vector3.Dot(hips.transform.up, Vector3.up);
        reward += upright * 0.01f;

        // 還站著就給一點生存獎勵
        if (hips.transform.position.y < 0.5f)
        {
            // 倒了
            AddReward(-1f);
            EndEpisode();
        }
        else
        {
            AddReward(reward);
        }
    }
}

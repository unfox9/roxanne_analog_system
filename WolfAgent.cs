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
        // 0: hips height (世界座標 y)
        sensor.AddObservation(hips.transform.position.y);

        // 1~3: hips 的「局部速度」(把世界速度轉到自身座標系)
        Vector3 velLocal = hips.transform.InverseTransformDirection(hips.velocity);
        sensor.AddObservation(velLocal);  // vx, vy, vz 3 個

        // 4~6: hips.transform.up (世界座標)
        Vector3 up = hips.transform.up;
        sensor.AddObservation(up);        // ux, uy, uz 3 個

        // 7~9: hips.transform.forward (世界座標)
        Vector3 fwd = hips.transform.forward;
        sensor.AddObservation(fwd);       // fx, fy, fz 3 個

        // 總共 = 1 + 3 + 3 + 3 = 10 個 float
    }


    public override void OnActionReceived(ActionBuffers actions)
    {
        var a = actions.ContinuousActions;
        int usable = Mathf.Min(joints.Length, a.Length);

        for (int i = 0; i < usable; i++)
        {
            float x = Mathf.Clamp(a[i], -1f, 1f);
            var jc = joints[i];
            var drive = jc.body.xDrive;

            // t 應該在 0~1 之間，不是 (x+1)*45
            float t = (x + 1f) * 0.5f; // -1~1 → 0~1
            float target = Mathf.Lerp(jc.minTarget, jc.maxTarget, t);

            drive.target = target;
            jc.body.xDrive = drive;
        }
    }

}

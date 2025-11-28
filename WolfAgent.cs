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

        // 這個關節用幾個 DOF?
        public bool useXDrive = true;   // Revolute / Spherical X
        public bool useYDrive = false;  // Spherical Y
        public bool useZDrive = false;  // Spherical Z

        // 每個 DOF 的角度範圍 (degree)
        public float minXTarget, maxXTarget;
        public float minYTarget, maxYTarget;
        public float minZTarget, maxZTarget;
    }

    [Header("Feet / Toes contacts")]
    public FootContact[] toeContacts;

    [Header("Body refs")]
    public ArticulationBody root;   // 角色最上層的 ArticulationBody
    public ArticulationBody hips;   // 主要質量中心 / 骨盆
    public JointControl[] joints;

    [Header("Decision")]
    public int decisionInterval = 5;
    int stepCount = 0;

    // 站立任務相關參數
    [Header("Standing Task")]
    public float minStandingHeight = 0.6f;   // 站立時大概的 y 高度 (可以視模型改)
    public float maxStandingHeight = 1.2f;   // 太高就不再額外加分
    public float fallHeightThreshold = 0.3f; // 低於這個就算跌倒
    public float minUprightDot = 0.3f;       // up.y 低於這個也算跌倒
    [Header("Reward Parameters")]
    public float rewardMinUpright = 0.01f;   // 對應 python 獎勵的 min_upright
    public float rewardComK = 5f;            // 對應 python 獎勵的 com_k
    public float rewardUprightWeight = 0.4f;
    public float rewardGroundWeight = 0.4f;
    public float rewardComWeight = 0.0f;     // 預設與 python 一樣關閉 COM 獎勵

    // 起始位置 (reset 用)
    Vector3 _startPos;
    Quaternion _startRot;

    // 用來畫 COM 的 cache
    ArticulationBody[] _bodies;

    void Awake()
    {
        if (hips != null)
        {
            _startPos = hips.transform.position;
            _startRot = hips.transform.rotation;
        }
    }

    Vector3 ComputeCOM()
    {
        CacheBodies();
        if (_bodies == null || _bodies.Length == 0)
        {
            if (hips != null) return hips.worldCenterOfMass;
            if (root != null) return root.transform.position;
            return Vector3.zero;
        }

        float totalMass = 0f;
        Vector3 com = Vector3.zero;
        foreach (var b in _bodies)
        {
            totalMass += b.mass;
            com += b.worldCenterOfMass * b.mass;
        }
        if (totalMass <= 0f)
        {
            if (hips != null) return hips.worldCenterOfMass;
            if (root != null) return root.transform.position;
            return Vector3.zero;
        }
        return com / totalMass;
    }

    void CacheBodies()
    {
        if (root != null && (_bodies == null || _bodies.Length == 0))
        {
            _bodies = root.GetComponentsInChildren<ArticulationBody>();
        }
    }

    void OnDrawGizmos()
    {
        if (root == null) return;

        // 直接用共用函式
        Vector3 com = ComputeCOM();

        Gizmos.color = Color.yellow;
        Gizmos.DrawSphere(com, 0.05f);
    }

    void Start()
    {
        // 幫你檢查「需要幾個連續動作」
        int expected = 0;
        foreach (var jc in joints)
        {
            if (jc.useXDrive) expected++;
            if (jc.useYDrive) expected++;
            if (jc.useZDrive) expected++;
        }

        Debug.Log($"[WolfAgent] Expected continuous actions = {expected}. " +
                  $"請在 BehaviorParameters 裡把 Action Size 設成 {expected} (Continuous).");
    }

    void FixedUpdate()
    {
        
        if (hips != null)
        {
            float hipY = hips.transform.position.y;
            float comY = ComputeCOM().y;
            //Debug.Log($"hip worldY = {hipY:F3}, COM worldY = {comY:F3}");
        }

        // 自己控制何時要 decision，不用 DecisionRequester
        stepCount++;
        if (stepCount % decisionInterval == 0)
        {
            RequestDecision();
        }
    }

    /*void ResetToStandPose()
    {
        if (hips != null)
        {
            hips.velocity = Vector3.zero;
            hips.angularVelocity = Vector3.zero;
            hips.transform.position = _startPos;
            hips.transform.rotation = _startRot;
        }

        foreach (var jc in joints)
        {
            var b = jc.body;
            if (b == null) continue;
            b.velocity = Vector3.zero;
            b.angularVelocity = Vector3.zero;

            var xd = b.xDrive;
            xd.target = 0f;
            b.xDrive = xd;

            var yd = b.yDrive;
            yd.target = 0f;
            b.yDrive = yd;

            var zd = b.zDrive;
            zd.target = 0f;
            b.zDrive = zd;
        }
    }:*/


    public override void OnEpisodeBegin()
    {
        //ResetToStandPose();
        stepCount = 0;
    }


    public override void CollectObservations(VectorSensor sensor)
    {
        if (hips == null)
        {
            // 設錯參考的話至少不會爆
            sensor.AddObservation(0f); // height
            sensor.AddObservation(Vector3.zero); // velLocal
            sensor.AddObservation(Vector3.up);   // up
            sensor.AddObservation(Vector3.forward); // forward
            return;
        }

    // ========= 1. root 資訊 (10) =============

    // height
    sensor.AddObservation(hips.transform.position.y);

    // local velocity
    Vector3 velLocal = hips.transform.InverseTransformDirection(hips.velocity);
    sensor.AddObservation(velLocal);

    // up / forward
    Vector3 up = hips.transform.up.normalized;
    Vector3 fwd = hips.transform.forward.normalized;

    sensor.AddObservation(up);
    sensor.AddObservation(fwd);

    // COM：用 hips 當 local 參考
    Vector3 comWorld = ComputeCOM();
    Vector3 comLocal = hips.transform.InverseTransformPoint(comWorld);

    sensor.AddObservation(comLocal);   // 這裡多 3 維

    // ========= 2. joints 資訊 (7 * joints.Count) ===========
        foreach (var jc in joints)
        {
            var b = jc.body;
            if (b == null)
            {
                // 補零保持維度固定
                sensor.AddObservation(new float[7]);
                continue;
            }

        Transform t = b.transform;

        // local rotation (相對 parent)
        Quaternion q = t.localRotation.normalized;
        sensor.AddObservation(q.x);
        sensor.AddObservation(q.y);
        sensor.AddObservation(q.z);
        sensor.AddObservation(q.w);

        // local angular velocity
        Vector3 angVelLocal = t.InverseTransformDirection(b.angularVelocity);
        sensor.AddObservation(angVelLocal);
        }

        // ========= 3. 腳趾接地 (toe contacts) ===========
        if (toeContacts != null)
        {
            foreach (var toe in toeContacts)
            {
                float grounded = (toe != null && toe.IsGrounded) ? 1f : 0f;
                sensor.AddObservation(grounded);
            }
        }
    }

    float ComputeGroundedFraction()
    {
        if (toeContacts == null || toeContacts.Length == 0)
            return 0f;

        float groundedCount = 0f;
        foreach (var toe in toeContacts)
        {
            if (toe != null && toe.IsGrounded)
            {
                groundedCount += 1f;
            }
        }

        return groundedCount / toeContacts.Length;
    }

    float ComputeReward()
    {
        if (hips == null)
            return 0f;

        Vector3 comLocal = hips.transform.InverseTransformPoint(ComputeCOM());
        Vector3 up = hips.transform.up.normalized;

        float upY = up.y;
        float comXY = new Vector2(comLocal.x, comLocal.y).magnitude;
        float groundedFrac = ComputeGroundedFraction();

        float rUpright = Mathf.Clamp01((upY - rewardMinUpright) / (1f - rewardMinUpright));
        float rCom = Mathf.Exp(-rewardComK * comXY * comXY);
        float rGround = groundedFrac;

        float reward = rewardUprightWeight * rUpright
                     + rewardComWeight * rCom
                     + rewardGroundWeight * rGround;

        return reward;
    }

    bool IsFallen()
    {
        if (hips == null) return false;

        // world-space COM
        Vector3 com = ComputeCOM();
        float height = com.y;

        if (height < fallHeightThreshold)
            return true;

        float upY = hips.transform.up.normalized.y;
        return upY < minUprightDot;
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        var a = actions.ContinuousActions;
        ApplyJointActions(a);

        float reward = ComputeReward();
        AddReward(reward);

        if (IsFallen())
        {
            EndEpisode();          // 建議：直接結束，下一回合再站好
        }
    }

    void ApplyJointActions(ActionSegment<float> a)
    {
        int actionIndex = 0;

        foreach (var jc in joints)
        {
            var body = jc.body;
            if (body == null) continue;

            // xDrive
            if (jc.useXDrive && actionIndex < a.Length)
            {
                float x = Mathf.Clamp(a[actionIndex++], -1f, 1f);
                float t = (x + 1f) * 0.5f;
                float target = Mathf.Lerp(jc.minXTarget, jc.maxXTarget, t);

                var drive = body.xDrive;
                drive.target = target;
                body.xDrive = drive;
            }

            // yDrive
            if (jc.useYDrive && actionIndex < a.Length)
            {
                float x = Mathf.Clamp(a[actionIndex++], -1f, 1f);
                float t = (x + 1f) * 0.5f;
                float target = Mathf.Lerp(jc.minYTarget, jc.maxYTarget, t);

                var drive = body.yDrive;
                drive.target = target;
                body.yDrive = drive;
            }

            // zDrive
            if (jc.useZDrive && actionIndex < a.Length)
            {
                float x = Mathf.Clamp(a[actionIndex++], -1f, 1f);
                float t = (x + 1f) * 0.5f;
                float target = Mathf.Lerp(jc.minZTarget, jc.maxZTarget, t);

                var drive = body.zDrive;
                drive.target = target;
                body.zDrive = drive;
            }

            if (actionIndex >= a.Length)
                break;
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // 如果想手動測試，可以在這裡塞鍵盤控制或 0
        var a = actionsOut.ContinuousActions;
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = 0f;
        }
    }
}

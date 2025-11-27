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
        // 自己控制何時要 decision，不用 DecisionRequester
        stepCount++;
        if (stepCount % decisionInterval == 0)
        {
            RequestDecision();
        }
    }

    void ResetToStandPose()
    {
        if (hips != null)
        {
            hips.transform.position = _startPos;
            hips.transform.rotation = _startRot;
            hips.velocity = Vector3.zero;
            hips.angularVelocity = Vector3.zero;
        }

        foreach (var jc in joints)
        {
            var b = jc.body;
            if (b == null) continue;

            b.velocity = Vector3.zero;
            b.angularVelocity = Vector3.zero;

            // 關節的 target 回到「站姿」
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
    }

    public override void OnEpisodeBegin()
    {
        ResetToStandPose();
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


    bool IsFallen()
    {
        if (hips == null) return false;

        float height = hips.transform.position.y;
        Vector3 up = hips.transform.up;
        float upDot = Vector3.Dot(up, Vector3.up);

        if (height < fallHeightThreshold) return true;
        if (upDot < minUprightDot) return true;

        return false;
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        var a = actions.ContinuousActions;
        ApplyJointActions(a);

        if (IsFallen())
        {
            EndEpisode();          // 建議：直接結束，下一回合再站好
            return;
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

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
        CacheBodies();
        if (_bodies == null || _bodies.Length == 0) return;

        float totalMass = 0f;
        Vector3 com = Vector3.zero;
        foreach (var b in _bodies)
        {
            totalMass += b.mass;
            com += b.worldCenterOfMass * b.mass;
        }
        if (totalMass <= 0f) return;

        com /= totalMass;

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

    public override void OnEpisodeBegin()
    {
        // Reset 位置 & 姿勢
        if (hips != null)
        {
            hips.transform.position = _startPos + new Vector3(
                Random.Range(-0.05f, 0.05f), 
                0.0f,
                Random.Range(-0.05f, 0.05f));
            hips.transform.rotation = _startRot;
            hips.velocity = Vector3.zero;
            hips.angularVelocity = Vector3.zero;
        }

        // 重設 joints
        foreach (var jc in joints)
        {
            var b = jc.body;
            if (b == null) continue;

            b.velocity = Vector3.zero;
            b.angularVelocity = Vector3.zero;

            // x 軸
            if (jc.useXDrive)
            {
                var drive = b.xDrive;
                drive.target = 0f;
                b.xDrive = drive;
            }
            // y 軸
            if (jc.useYDrive)
            {
                var drive = b.yDrive;
                drive.target = 0f;
                b.yDrive = drive;
            }
            // z 軸
            if (jc.useZDrive)
            {
                var drive = b.zDrive;
                drive.target = 0f;
                b.zDrive = drive;
            }
        }

        // 歸零 step
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

        // 0: hips height (世界座標 y)
        sensor.AddObservation(hips.transform.position.y);

        // 1~3: hips 的「局部速度」
        Vector3 velLocal = hips.transform.InverseTransformDirection(hips.velocity);
        sensor.AddObservation(velLocal);  // vx, vy, vz

        // 4~6: hips.transform.up
        Vector3 up = hips.transform.up.normalized;
        sensor.AddObservation(up);        // ux, uy, uz

        // 7~9: hips.transform.forward
        Vector3 fwd = hips.transform.forward.normalized;
        sensor.AddObservation(fwd);       // fx, fy, fz

        // 目前總共 10 個 float
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        var a = actions.ContinuousActions;
        ApplyJointActions(a);

        if (hips == null)
            return;

        // --------- 計算狀態量 (這裡重新算，不吃 CollectObservations 的變數) ---------
        float height = hips.transform.position.y;
        Vector3 up = hips.transform.up.normalized;
        Vector3 velLocal = hips.transform.InverseTransformDirection(hips.velocity);

        // --------- Safety：跌倒就結束 ---------
        if (height < fallHeightThreshold || up.y < minUprightDot)
        {
            // 跌倒可以給一個小 penalty（避免亂姿勢）
            AddReward(-1.0f);
            EndEpisode();
            return;
        }

        // --------- Reward 設計：目標 = 站好 ---------
        float reward = 0f;

        // (1) 高度越接近合理站立高度區間越好
        float heightNorm = Mathf.InverseLerp(minStandingHeight, maxStandingHeight, Mathf.Clamp(height, minStandingHeight, maxStandingHeight));
        // 在區間內：0~1 之間；低於 minStandingHeight 就 0；高於 maxStandingHeight 也當 1
        reward += 0.4f * heightNorm;

        // (2) 身體越直立 (up.y 越接近 1) 越好
        // up.y 本身就大概是 -1 ~ 1，站好大約 0.7~1
        float uprightReward = Mathf.InverseLerp(0.0f, 1.0f, Mathf.Clamp01(up.y));
        reward += 0.5f * uprightReward;

        // (3) 不亂晃：水平速度越小越好
        Vector2 horizVel = new Vector2(velLocal.x, velLocal.z);
        float horizSpeed = horizVel.magnitude;
        // 小扣分：速度越大越扣
        reward -= 0.05f * horizSpeed;

        // (4) 活著就有一點活躍獎勵，鼓勵長時間維持站姿
        reward += 0.001f;

        // 用 fixedDeltaTime 做時間尺度無關的 reward
        AddReward(reward * Time.fixedDeltaTime);
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

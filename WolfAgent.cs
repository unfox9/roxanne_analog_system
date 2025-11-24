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

        //這個關節用幾個DOF?
        public bool useXDrive = true;    // xDrive:Revolute
        public bool useYDrive = false;  // spherical 的 YDrive
        public bool useZDrive = false;  // spherical 的 ZDrive

        //每個DOF的角度範圍(degree)
        public float minXTarget, maxXTarget;
        public float minYTarget, maxYTarget;
        public float minZTarget, maxZTarget;
    }

    public ArticulationBody hips;
    public JointControl[] joints;


    public ArticulationBody root;  // 在 Inspector 把角色最上層有 ArticulationBody 的丟進來

    ArticulationBody[] _bodies;

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

        Gizmos.DrawSphere(com, 0.05f);
    }


    public int decisionInterval;
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
        // reset 姿勢 & 速度
        hips.velocity = Vector3.zero;
        hips.angularVelocity = Vector3.zero;
        
        foreach (var jc in joints)
        {
            var b = jc.body;
            if (b == null) continue;

            // 清關節剛體的速度
            b.velocity = Vector3.zero;
            b.angularVelocity = Vector3.zero;

            // x 軸
            if (jc.useXDrive)
            {
                var drive = b.xDrive;
                drive.target = 0;
                b.xDrive = drive;
            }

            // y 軸
            if (jc.useYDrive)
            {
                var drive = b.yDrive;
                drive.target = 0;
                b.yDrive = drive;
            }

            // z 軸
            if (jc.useZDrive)
            {
                var drive = b.zDrive;
                drive.target = 0;
                b.zDrive = drive;
            }
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // hips 高度
        sensor.AddObservation(hips.transform.position.y);

        // hips 局部速度
        sensor.AddObservation(hips.transform.InverseTransformDirection(hips.velocity));

        // hips up / forward
        sensor.AddObservation(hips.transform.up);
        sensor.AddObservation(hips.transform.forward);

        // hips 局部角速度
        sensor.AddObservation(hips.transform.InverseTransformDirection(hips.angularVelocity));

        // 左右腳接地
        sensor.AddObservation(leftFootContact ? 1f : 0f);
        sensor.AddObservation(rightFootContact ? 1f : 0f);


        // 遍歷每個關節
        foreach (var jc in joints)
        {
            ArticulationBody b = jc.body;
            if (b == null) continue;

                int dof = b.jointPosition.dofCount;

                for (int i = 0; i < dof; i++)
                {
                float angleRad = b.jointPosition[i];
                float velRad = b.jointVelocity[i] * 0.1f;  // scaled

                float minRad, maxRad;
                float normalizedAngle = 0f;

                if (i == 0 && jc.useXDrive)
                {
                    minRad = jc.minXTarget * Mathf.Deg2Rad;
                    maxRad = jc.maxXTarget * Mathf.Deg2Rad;
                    normalizedAngle = Mathf.InverseLerp(minRad, maxRad, angleRad) * 2f - 1f;
                }
                else if (i == 1 && jc.useYDrive)
                {
                    minRad = jc.minYTarget * Mathf.Deg2Rad;
                    maxRad = jc.maxYTarget * Mathf.Deg2Rad;
                    normalizedAngle = Mathf.InverseLerp(minRad, maxRad, angleRad) * 2f - 1f;
                }
                else if (i == 2 && jc.useZDrive)
                {
                    minRad = jc.minZTarget * Mathf.Deg2Rad;
                    maxRad = jc.maxZTarget * Mathf.Deg2Rad;
                    normalizedAngle = Mathf.InverseLerp(minRad, maxRad, angleRad) * 2f - 1f;
                }
                else
                {
                    continue;
                }

                sensor.AddObservation(normalizedAngle);
                sensor.AddObservation(velRad);
            }
        }
    }



    public override void OnActionReceived(ActionBuffers actions)
    {
        var a = actions.ContinuousActions;
        int actionIndex = 0;

        foreach (var jc in joints)
        {
            var body = jc.body;
            if (body == null) continue;
            if (actionIndex >= a.Length) break; // 沒 action 了就停
            

            // Twist / xDrive (Revolute/Spherical)
            if (jc.useXDrive)
            {
                float x = Mathf.Clamp(a[actionIndex++], -1f, 1f);
                float t = (x + 1f) * 0.5f;
                float target = Mathf.Lerp(jc.minXTarget, jc.maxXTarget, t);


                var drive = body.xDrive;
                drive.target = target;
                body.xDrive = drive;
            }


            // yDrive (Spherical)
            if (jc.useYDrive && actionIndex < a.Length)
            {
                float x = Mathf.Clamp(a[actionIndex++], -1f, 1f);
                float t = (x + 1f) * 0.5f;
                float target = Mathf.Lerp(jc.minYTarget, jc.maxYTarget, t);


                var drive = body.yDrive;
                drive.target = target;
                body.yDrive = drive;
            }


            // zDrive (Spherical)
            if (jc.useZDrive && actionIndex < a.Length)
            {
                float x = Mathf.Clamp(a[actionIndex++], -1f, 1f);
                float t = (x + 1f) * 0.5f;
                float target = Mathf.Lerp(jc.minZTarget, jc.maxZTarget, t);


                var drive = body.zDrive;
                drive.target = target;
                body.zDrive = drive;
            }
        }
    }

    void Start()
    {
        int expected = 0;
        foreach (var jc in joints)
        {
            if (jc.useXDrive) expected++;
            if (jc.useYDrive) expected++;
            if (jc.useZDrive) expected++;
        }

        Debug.Log($"[WolfAgent] Excepted continuous actions = {expected}");
    }

}

using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;
using Unity.VisualScripting;

public class WolfAgent : Agent
{
    [System.Serializable]
    public class JointControl
    {
        public ArticulationBody body;

        public bool useXDrive = true;   
        public bool useYDrive = false; 
        public bool useZDrive = false;  

        public float minXTarget, maxXTarget;
        public float minYTarget, maxYTarget;
        public float minZTarget, maxZTarget;
    }

    [Header("Feet / Toes contacts")]
    public FootContact[] toeContacts;

    [Header("Body refs")]
    public ArticulationBody root;   
    public ArticulationBody hips;  
    public JointControl[] joints;

    [Header("Decision")]
    public int decisionInterval = 1;
    int stepCount = 0;

    Vector3 _startPos;
    Quaternion _startRot;
    Vector3 _rootStartPos;
    Quaternion _rootStartRot;

    ArticulationBody[] _bodies;

    void Start()
    {
        if (root != null)
        {
            _rootStartPos = root.transform.position;
            _rootStartRot = root.transform.rotation;
        }
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
        Vector3 com = ComputeCOM();
        Gizmos.color = Color.yellow;
        Gizmos.DrawSphere(com, 0.05f);
    }

    void Awake()
    {
        int expected = 0;
        foreach (var jc in joints)
        {
            if (jc.useXDrive) expected++;
            if (jc.useYDrive) expected++;
            if (jc.useZDrive) expected++;
        }

        Debug.Log($"[WolfAgent] Expected continuous actions = {expected}. " );
    }

    void FixedUpdate()
    {
        stepCount++;
        if (stepCount % decisionInterval == 0)
        {
            RequestDecision();
        }
    }

    void ResetToStandPose()
    {
        if (root != null)
        {
            root.TeleportRoot(_rootStartPos, _rootStartRot);
            root.velocity = Vector3.zero;
            root.angularVelocity = Vector3.zero;
            root.jointVelocity = new ArticulationReducedSpace(0f, 0f, 0f);
        }
        else if (hips != null)
        {
            hips.TeleportRoot(_startPos, _startRot);
            hips.velocity = Vector3.zero;
            hips.angularVelocity = Vector3.zero;
            hips.jointVelocity = new ArticulationReducedSpace(0f, 0f, 0f);
        }

        foreach (var jc in joints)
        {
            var b = jc.body;
            if (b == null) continue;

            b.jointPosition = new ArticulationReducedSpace(0f, 0f, 0f);
            b.jointVelocity = new ArticulationReducedSpace(0f, 0f, 0f);

            b.velocity = Vector3.zero;
            b.angularVelocity = Vector3.zero;

            if (jc.useXDrive)
            {
                var xd = b.xDrive;
                xd.target = 0f;
                b.xDrive = xd;
            }

            if (jc.useYDrive)
            {
                var yd = b.yDrive;
                yd.target = 0f;
                b.yDrive = yd;
            }

            if (jc.useZDrive)
            {
                var zd = b.zDrive;
                zd.target = 0f;
                b.zDrive = zd;
            }
        }
    
        Physics.SyncTransforms();
    }


    public override void OnEpisodeBegin()
    {
        ResetToStandPose();
        stepCount = 0;

        if (_prevAction != null)
        {
            for (int i = 0; i < _prevAction.Length; i++)
                _prevAction[i] = 0f;
        }

        _prevSupportDist = 0f;
        _hasPrevSupport = false;
        _hasPrevAction = false;
    }

    int GetObsSize()
    {
        int rootDim = 13;
        int jointDim = 6 * joints.Length; 
        int toeDim = toeContacts != null ? toeContacts.Length : 0;
        return rootDim + jointDim + toeDim;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (hips == null)
        {
            int dim = GetObsSize();
            for (int i = 0; i < dim; i++)
                sensor.AddObservation(0f);
            return;
        }

    float maxHipHeight = 2f;
    float hipYNorm = Mathf.Clamp01(hips.transform.position.y / maxHipHeight);
    sensor.AddObservation(hipYNorm);

    Vector3 velLocal = hips.transform.InverseTransformDirection(hips.velocity);
    Vector3 velNorm = Vector3.ClampMagnitude(velLocal / 10f, 1f);
    sensor.AddObservation(velNorm);

    Vector3 up = hips.transform.up.normalized;
    Vector3 fwd = hips.transform.forward.normalized;
    sensor.AddObservation(up);
    sensor.AddObservation(fwd);

    Vector3 comWorld = ComputeCOM();
    Vector3 comLocal = hips.transform.InverseTransformPoint(comWorld);
    Vector3 comNorm = Vector3.ClampMagnitude(comLocal / 2f, 1f);
    sensor.AddObservation(comNorm);

        foreach (var jc in joints)
        {
            AddJointObs(sensor, jc);
        }
    
        if (toeContacts != null)
        {
            foreach (var toe in toeContacts)
            {
                float grounded = (toe != null && toe.IsGrounded) ? 1f : 0f;
                sensor.AddObservation(grounded);
            }
        }
    }

    float NormalizeAngle(float angleDeg, float minDeg, float maxDeg)
    {
        float center = 0.5f * (minDeg + maxDeg);
        float halfRange = 0.5f * (maxDeg - minDeg);
        if (halfRange < 1e-3f) return 0f;
        return Mathf.Clamp((angleDeg - center) / halfRange, -1f, 1f);
    }

    void AddJointObs(VectorSensor sensor, JointControl jc)
    {
        var b = jc.body;
        if (b == null)
        {
            sensor.AddObservation(new float[6]);
            return;
        }

        var jp = b.jointPosition;
        var jv = b.jointVelocity;
        int dof = b.dofCount; 
        
        float xDeg = (dof > 0) ? jp[0] * Mathf.Rad2Deg : 0f;
        float vx = (dof > 0) ? jv[0] : 0f;
        float xNorm = NormalizeAngle(xDeg, jc.minXTarget, jc.maxXTarget);
        float vxNorm = Mathf.Clamp(vx / 20f, -1f, 1f);

        float yDeg = (dof > 1) ? jp[1] * Mathf.Rad2Deg : 0f;
        float vy = (dof > 1) ? jv[1] : 0f;
        float yNorm = NormalizeAngle(yDeg, jc.minYTarget, jc.maxYTarget);
        float vyNorm = Mathf.Clamp(vy / 20f, -1f, 1f);

        float zDeg = (dof > 2) ? jp[2] * Mathf.Rad2Deg : 0f;
        float vz = (dof > 2) ? jv[2] : 0f;
        float zNorm = NormalizeAngle(zDeg, jc.minZTarget, jc.maxZTarget);
        float vzNorm = Mathf.Clamp(vz / 20f, -1f, 1f);

        sensor.AddObservation(xNorm); // Normalized
        sensor.AddObservation(yNorm);
        sensor.AddObservation(zNorm);
        sensor.AddObservation(vxNorm);
        sensor.AddObservation(vyNorm);
        sensor.AddObservation(vzNorm);
    }

    [Header("Ground / Support")]
    public float groundSupportK = 10f;
    float _prevSupportDist = 0f;
    bool _hasPrevSupport = false;
    
    float ComputeGroundSupport(Vector3 comWorld, out float currentDist)
    {
        currentDist = 0f;

        if (toeContacts == null || toeContacts.Length == 0 || hips == null)
            return 0f;

        Vector2 supportCenter = Vector2.zero;
        int groundedCount = 0;

        foreach (var toe in toeContacts)
        {
            if (toe != null && toe.IsGrounded)
            {
                groundedCount++;

                Vector3 toeLocal = hips.transform.InverseTransformPoint(toe.transform.position);
                supportCenter += new Vector2(toeLocal.x, toeLocal.z);
            }
        }

        if (groundedCount == 0)
            return 0f;
        
        supportCenter /= groundedCount;

        Vector3 comLocal = hips.transform.InverseTransformPoint(comWorld);
        Vector2 comXZ = new Vector2(comLocal.x, comLocal.z);

        float dist = (comXZ - supportCenter).magnitude;
        currentDist = dist;

        float align = Mathf.Exp(-groundSupportK * dist * dist);
        float groundedFrac = (float)groundedCount / toeContacts.Length;

        return (groundedFrac * 0.5f + 0.5f) * align;
    }

    [Header("Standing Task")]
    public float minStandingHeight = 0.6f;   
    public float maxStandingHeight = 1.2f;   
    public float fallHeightThreshold = 0.3f; 
    public float minUprightDot = 0.3f;       
    public float fallPenalty = 10f;          
    [Header("Reward Parameters")]
    public float rewardMinUpright = 0.01f;   
    public float rewardComK = 5f;            
    public float rewardUprightWeight = 0.4f;
    public float rewardGroundWeight = 0.4f;
    public float rewardComWeight = 0.4f;     
    [Header("Stability / Velocity")]
    public float velK = 0.5f;         
    public float rewardVelocityWeight = 0.2f;
    [Header("Height Reward")]
    public float rewardHeightWeight = 0.2f;
    [Header("Alive Bonus")]
    public float aliveBonus = 0.01f; 
        
    float ComputeReward()
    {
        if (hips == null)
            return 0f;

        Vector3 up = hips.transform.up.normalized;
        float upDot = Vector3.Dot(up, Vector3.up);

        Vector3 comWorld = ComputeCOM();
        Vector3 comLocal = hips.transform.InverseTransformPoint(comWorld);
        float comXY = new Vector2(comLocal.x, comLocal.z).magnitude;

        float rUpright = Mathf.Clamp01(
            (upDot - rewardMinUpright) / (1f - rewardMinUpright)
        );

        float rCom = Mathf.Exp(-rewardComK * comXY * comXY);

        float supportDist;
        float baseGround = ComputeGroundSupport(comWorld, out supportDist);

        float rGround = baseGround;
        if (_hasPrevSupport)
        {
            float delta = _prevSupportDist - supportDist;
            if (delta > 0f)
            {
                float improvement = Mathf.Clamp(delta, 0f, 0.1f);
                rGround += improvement * 5f;
            }
        }

        _prevSupportDist = supportDist;
        _hasPrevSupport = true;

        float hipY = hips.transform.position.y;
        float hNorm = Mathf.InverseLerp(minStandingHeight, maxStandingHeight, hipY);
        float rHeight = Mathf.Clamp01(hNorm);

        float rVel = 1f;
        if (upDot > 0.5f && hipY > minStandingHeight)
        {
            Vector3 vel = hips.velocity;
            Vector2 velXZ = new Vector2(vel.x, vel.z);
            rVel = Mathf.Exp(-velK * velXZ.sqrMagnitude);
        }

        float reward = rewardUprightWeight * rUpright
                     + rewardComWeight * rCom
                     + rewardGroundWeight * rGround
                     + rewardVelocityWeight * rVel
                     + rewardHeightWeight * rHeight;

        reward *= 0.3f;
        return reward;
    }

    bool IsFallen()
    {

        if (hips == null)
        return true;

        float hipY = hips.transform.position.y;
        if (hipY < fallHeightThreshold)
            return true;

        
        float upDot = Vector3.Dot(hips.transform.up.normalized, Vector3.up);

        if (upDot < minUprightDot)
            return true;

        return false;
    }

    [Header("Action Regularization")]
    public float actionL2Weight = 0.001f;
    public float actionSmoothWeight = 0.001f;
    float[] _prevAction;
    bool _hasPrevAction;

    public override void OnActionReceived(ActionBuffers actions)
    {
        var a = actions.ContinuousActions;

        if (_prevAction == null || _prevAction.Length != a.Length)
            _prevAction = new float[a.Length];

        ApplyJointActions(a);

        if (IsFallen())
        {
            AddReward(-fallPenalty); 
            EndEpisode();        
            return;
        }

        float reward = ComputeReward();

        float actL2 = 0f;
        float smooth = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            float v = a[i];
            actL2 += v * v;
            if (_hasPrevAction)
            {
                float diff = v - _prevAction[i];
                smooth += diff * diff;
            }
            _prevAction[i] = v;
        }
        _hasPrevAction = true;

        actL2 /= a.Length;
        smooth /= a.Length;
        float regPenalty = 
            actionL2Weight * actL2 + 
            actionSmoothWeight * smooth;

        reward -= regPenalty;

        AddReward(reward);

        if (hips.transform.position.y > minStandingHeight &&
            Vector3.Dot(hips.transform.up, Vector3.up) > minUprightDot)
        {
            AddReward(aliveBonus);
        }
    }

    void ApplyJointActions(ActionSegment<float> a)
    {
        int actionIndex = 0;

        foreach (var jc in joints)
        {
            var body = jc.body;
            if (body == null) continue;

            if (jc.useXDrive && actionIndex < a.Length)
            {
                float x = Mathf.Clamp(a[actionIndex++], -1f, 1f);
                float t = (x + 1f) * 0.5f;
                float target = Mathf.Lerp(jc.minXTarget, jc.maxXTarget, t);

                var drive = body.xDrive;
                drive.target = target;
                body.xDrive = drive;
            }

            if (jc.useYDrive && actionIndex < a.Length)
            {
                float x = Mathf.Clamp(a[actionIndex++], -1f, 1f);
                float t = (x + 1f) * 0.5f;
                float target = Mathf.Lerp(jc.minYTarget, jc.maxYTarget, t);

                var drive = body.yDrive;
                drive.target = target;
                body.yDrive = drive;
            }

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
        var a = actionsOut.ContinuousActions;
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = 0f;
        }
    }
}
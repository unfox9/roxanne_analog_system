using UnityEngine;

public class FootContact : MonoBehaviour
{
    // 有多少東西正在踩著 / 接觸這個腳趾
    int _contactCount = 0;

    // 給別人讀的屬性：有沒有接地
    public bool IsGrounded => _contactCount > 0;

    void OnCollisionEnter(Collision collision)
    {
        // 忽略 trigger，只算實體碰撞
        if (!collision.collider.isTrigger)
        {
            _contactCount++;
        }
    }

    void OnCollisionExit(Collision collision)
    {
        if (!collision.collider.isTrigger)
        {
            _contactCount--;
            if (_contactCount < 0) _contactCount = 0; // 防呆
        }
    }

    // 如果你用的是 isTrigger 的 Collider，可以改用這個版本：
    /*
    void OnTriggerEnter(Collider other)
    {
        if (!other.isTrigger)
        {
            _contactCount++;
        }
    }

    void OnTriggerExit(Collider other)
    {
        if (!other.isTrigger)
        {
            _contactCount--;
            if (_contactCount < 0) _contactCount = 0;
        }
    }
    */
}

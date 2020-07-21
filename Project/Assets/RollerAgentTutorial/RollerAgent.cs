using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

namespace RollerAgentTutorial {

public class RollerAgent : Agent {
    public Transform target;
    public float forceMultiplier = 10;

    private Rigidbody _rb;

    private void Start () {
        this._rb = GetComponent<Rigidbody>();
    }

    override public void OnEpisodeBegin () {
        if (this.transform.localPosition.y < 0) {
            // If the Agent fell, zero its momentum
            this._rb.angularVelocity = Vector3.zero;
            this._rb.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        // Move the target to a new spot
        this.target.localPosition = new Vector3(
            Random.value * 8 - 4,
            0.5f,
            Random.value * 8 - 4
        );
    }

    override public void CollectObservations (VectorSensor sensor) {
        // Target and agent positions
        sensor.AddObservation(this.target.localPosition);
        sensor.AddObservation(this.transform.localPosition);

        // Agent velocity
        sensor.AddObservation(this._rb.velocity.x);
        sensor.AddObservation(this._rb.velocity.z);
    }

    override public void OnActionReceived (float[] vectorAction) {
        // Actions (size = 2)
        var controlSignal = Vector3.zero;
        controlSignal.x = vectorAction[0];
        controlSignal.z = vectorAction[1];
        this._rb.AddForce(controlSignal * this.forceMultiplier);

        // Rewards
        var distanceToTarget =
            Vector3.Distance(this.transform.localPosition, this.target.localPosition);

        if (this.transform.localPosition.y < 0) {
            EndEpisode();
        } else if (distanceToTarget < 1.42f) {
            SetReward(1f);
            EndEpisode();
        }
    }

    override public void Heuristic (float[] actionsOut) {
        actionsOut[0] = Input.GetAxis("Horizontal");
        actionsOut[1] = Input.GetAxis("Vertical");
    }
}

}

; =============================================================================
; PDDL domain: PushT with obstacles (tabletop manipulation)
; =============================================================================
; TABLE: Grid of regions r_m_n (m=row, n=column). Goal = one of these regions.
; BLOCKS: 10 obstacle blocks (block0..block9). Each block has a SIZE relative to
;   the gripper/end-effector:
;   - (pickable ?b): block is SMALLER than gripper → use PICK and PLACE to move
;     it out of the way.
;   - (push-only ?b): block is BIGGER than gripper → cannot pick; use PUSH_BLOCK
;     to push it one cell out of the way.
; The planner must figure out from (pickable blocki) and (push-only blockj) which
; action to use for each obstacle. Main objective: get the tee to the goal region.
; =============================================================================

(define (domain pusht)
  (:requirements :strips :typing)
  (:types robot tee block region)

  (:predicates
    (robot-at ?r - robot ?loc - region)
    (object-at ?o - tee ?loc - region)
    (block-at ?b - block ?loc - region)
    (holding ?r - robot ?b - block)
    (goal-at ?loc - region)
    (clear ?loc - region)
    (adjacent ?loc1 ?loc2 - region)
    (pickable ?b - block)
    (push-only ?b - block))

  (:action move_ee
    :parameters (?rob - robot ?from ?to - region)
    :precondition (and (robot-at ?rob ?from) (adjacent ?from ?to) (clear ?to))
    :effect (and (robot-at ?rob ?to) (not (robot-at ?rob ?from))))

  (:action push_tee
    :parameters (?rob - robot ?from ?to - region)
    :precondition (and (robot-at ?rob ?from) (object-at tee ?from) (adjacent ?from ?to) (clear ?to))
    :effect (and (object-at tee ?to) (not (object-at tee ?from)) (robot-at ?rob ?to) (not (robot-at ?rob ?from))))

  (:action pick
    :parameters (?rob - robot ?b - block ?loc - region)
    :precondition (and (pickable ?b) (robot-at ?rob ?loc) (block-at ?b ?loc) (not (holding ?rob ?b)))
    :effect (and (holding ?rob ?b) (not (block-at ?b ?loc)) (clear ?loc)))

  (:action place
    :parameters (?rob - robot ?b - block ?loc - region)
    :precondition (and (holding ?rob ?b) (robot-at ?rob ?loc) (clear ?loc))
    :effect (and (block-at ?b ?loc) (not (holding ?rob ?b)) (not (clear ?loc))))

  (:action push_block
    :parameters (?rob - robot ?b - block ?from ?to - region)
    :precondition (and (push-only ?b) (block-at ?b ?from) (robot-at ?rob ?from) (adjacent ?from ?to) (clear ?to))
    :effect (and (block-at ?b ?to) (not (block-at ?b ?from)) (robot-at ?rob ?to) (not (robot-at ?rob ?from)) (clear ?from) (not (clear ?to)))))

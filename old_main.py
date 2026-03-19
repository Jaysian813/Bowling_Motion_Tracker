import os
import cv2
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
import ffmpeg
from mmpose.apis import MMPoseInferencer
from scipy.signal import find_peaks
from tqdm import tqdm

class RTMPoseBowlingPipeline:
    """
    RTMPose 네이티브 버전을 사용한 투핸드 볼링 폼 분석 및 렌더링 파이프라인.
    """
    
    # RTMPose (COCO 17-keypoints) 인덱스 상수 정의
    KP = {
        'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
        'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
    }

    def __init__(self, pose2d_model='rtmpose-m_8xb256-420e_coco-256x192', font_path='/System/Library/Fonts/Supplemental/Arial.ttf'):
        print("Initializing MMPoseInferencer...")
        self.inferencer = MMPoseInferencer(pose2d=pose2d_model)
        self.font_path = font_path

    def calculate_angle(self, a, b, c):
        """세 점 a, b, c (각 점은 [x,y])를 받아 b를 꼭지점으로 하는 각도(도)를 구합니다."""
        ba = [a[0] - b[0], a[1] - b[1]]
        bc = [c[0] - b[0], c[1] - b[1]]
        dot_prod = ba[0]*bc[0] + ba[1]*bc[1]
        mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
        mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
        if mag_ba == 0 or mag_bc == 0:
            return 0
        cosine_angle = dot_prod / (mag_ba * mag_bc)
        # float 오차 보정
        cosine_angle = max(-1.0, min(1.0, cosine_angle))
        angle = math.degrees(math.acos(cosine_angle))
        return angle

    def extract_keypoints(self, video_path):
        """비디오에서 프레임별 키포인트를 추출하여 딕셔너리로 반환합니다."""
        print(f"Extracting pose data from: {video_path}")
        
        # 1. Get initial estimate
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        pose_data = {}
        results = self.inferencer(video_path, return_vis=False)
        
        # 2. Initialize pbar without a strict total if it might be wrong
        with tqdm(total=total_frames, desc="Bowling Pose Analysis") as pbar:
            for frame_idx, res in enumerate(results):
                preds = res.get('predictions', [[]])
                
                if preds and len(preds[0]) > 0:
                    kpts = preds[0][0].get('keypoints')
                    if kpts:
                        pose_data[frame_idx] = kpts
                
                pbar.update(1) # Manually move forward by 1
            
            # 3. Force the bar to 100% based on whatever frames were actually processed
            pbar.n = pbar.total = (frame_idx + 1)
            pbar.refresh()

        print("\nBowling Pose Analysis Complete")
        return pose_data

    def segment_phases(self, pose_data, dominant_hand='right'):
        """속도 및 역추적(Backtracking)을 활용한 단계 분리 (RTMPose 배열 구조용)"""
        frames = sorted(pose_data.keys())
        if len(frames) < 10:
            return {}

        wrist_idx = self.KP[f'{dominant_hand}_wrist']
        
        # 데이터 정리 및 스무딩 (Moving Average)
        raw_wrist_y = []
        raw_wrist_x = []
        raw_body_x = []
        valid_frames = []
        for f in frames:
            kpt = pose_data[f][wrist_idx]
            raw_wrist_y.append(kpt[1])
            raw_wrist_x.append(kpt[0])
            
            # 몸의 중심(골반 평균) X 좌표 추출
            l_hip = pose_data[f][self.KP['left_hip']]
            r_hip = pose_data[f][self.KP['right_hip']]
            raw_body_x.append((l_hip[0] + r_hip[0]) / 2.0)
            
            valid_frames.append(f)
            
        def moving_average(data, window_size=5):
            arr = np.array(data)
            padded = np.pad(arr, (window_size//2, window_size-1 - window_size//2), mode='edge')
            return np.convolve(padded, np.ones(window_size)/window_size, mode='valid')

        smoothed_y = moving_average(raw_wrist_y, window_size=5)
        smoothed_x = moving_average(raw_wrist_x, window_size=5)
        smoothed_body_x = moving_average(raw_body_x, window_size=5)
        
        smoothed_coords = {valid_frames[i]: (smoothed_x[i], smoothed_y[i]) for i in range(len(valid_frames))}
        smoothed_speeds = {}
        
        for i in range(len(valid_frames)):
            if i == 0:
                smoothed_speeds[valid_frames[i]] = 0
            else:
                dy = smoothed_y[i] - smoothed_y[i-1]
                dx = smoothed_x[i] - smoothed_x[i-1]
                smoothed_speeds[valid_frames[i]] = math.hypot(dx, dy)

        # 1. 릴리즈 포인트 (속도 정점 근처의 최하점)
        search_range = valid_frames[int(len(valid_frames)*0.4):int(len(valid_frames)*0.9)]
        if not search_range: search_range = valid_frames
        release_peak_speed_frame = max(search_range, key=lambda f: smoothed_speeds[f])
        
        rel_window = [f for f in valid_frames if abs(f - release_peak_speed_frame) < 15]
        release_f = max(rel_window, key=lambda f: smoothed_coords[f][1])

        # 2. 백 스윙 정점 (릴리즈 직전의 로컬 최상점)
        pre_release_idx = valid_frames.index(release_f)
        backswing_f = release_f
        found_ascent = False
        
        for i in range(pre_release_idx - 1, 0, -1):
            f_curr = valid_frames[i]
            f_next = valid_frames[i+1]
            if smoothed_coords[f_curr][1] < smoothed_coords[f_next][1]:
                found_ascent = True
                backswing_f = f_curr
            else:
                if found_ascent: # 올라가다 꺾이는 지점(고점) 발견
                    break

        # 3. 백 스윙 시작 (Pushaway 끝, Valley)
        bs_idx = valid_frames.index(backswing_f)
        valley_f = backswing_f
        found_valley = False
        
        for i in range(bs_idx - 1, 0, -1):
            f_curr = valid_frames[i]
            f_next = valid_frames[i+1]
            if smoothed_coords[f_curr][1] > smoothed_coords[f_next][1]: # 높이가 낮아짐 (Y 증가)
                found_valley = True
                valley_f = f_curr
            else:
                if found_valley: # 계곡 바닥을 지나 다시 올라가면(스탠스 쪽으로) 중단
                    break

        # 4. 푸시 어웨이 시작 지점
        # 스탠스(처음 5프레임)의 평균 손목 높이 및 몸통 위치 계산
        base_y = np.mean([smoothed_coords[valid_frames[i]][1] for i in range(min(5, len(valid_frames)))])
        base_body_x = np.mean([smoothed_body_x[i] for i in range(min(5, len(valid_frames)))])
        
        # --- 몸통 길이(참조 길이) 계산 ---
        ref_length = 100.0  # 기본값 fallback
        try:
            fkpts = pose_data[valid_frames[0]]
            ls, rs = fkpts[KP['left_shoulder']], fkpts[KP['right_shoulder']]
            lh, rh = fkpts[KP['left_hip']], fkpts[KP['right_hip']]
            mid_s = [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2]
            mid_h = [(lh[0]+rh[0])/2, (lh[1]+rh[1])/2]
            ref_length = max(50.0, np.linalg.norm(np.array(mid_s) - np.array(mid_h)))
        except:
            pass
            
        pushaway_start_f = valid_frames[0]
        valley_idx = valid_frames.index(valley_f)
        
        for i in range(valley_idx):
            f_curr = valid_frames[i]
            # 평균보다 손목이 '몸통 길이의 10%' 이상 내려가고, 골반이 '몸통 길이의 5%' 이상 이동했을 때 푸시어웨이 시작
            wrist_dropped = smoothed_coords[f_curr][1] > base_y + (ref_length * 0.10)
            body_moved = abs(smoothed_body_x[i] - base_body_x) > (ref_length * 0.05)
            
            if wrist_dropped and body_moved:
                pushaway_start_f = f_curr
                break
                
        # 만약 찾지 못했다면 계곡 지점 조금 앞으로 임의 설정
        if pushaway_start_f == valid_frames[0] and valley_idx > 5:
            pushaway_start_f = valid_frames[max(0, valley_idx - 10)]

        # 5. 스탠스는 시작 부분
        stance_f = valid_frames[0]

        # 6. 팔로우스루는 끝 부분
        ft_f = valid_frames[-1]

        return {
            'stance': stance_f,
            'pushaway_start': pushaway_start_f,
            'backswing_start': valley_f,
            'backswing_top': backswing_f,
            'release': release_f,
            'follow_through_end': ft_f
        }

    def _evaluate_metric(self, name, description, actual_val, ideal_min, ideal_max, penalty_weight, unit="°"):
        """평가 및 점수 감점 계산 유틸리티"""
        if actual_val is None:
            return {"name": name, "description": description, "status": "error", "deduction": 0.0}
            
        penalty = 0.0
        status = "perfect"
        
        if ideal_min is not None and ideal_max is not None:
            ideal_str = f"{ideal_min}~{ideal_max} {unit}"
            if actual_val < ideal_min:
                penalty = (ideal_min - actual_val) * penalty_weight
                status = "needs_improvement"
            elif actual_val > ideal_max:
                penalty = (actual_val - ideal_max) * penalty_weight
                status = "needs_improvement"
        elif ideal_min is not None:
            ideal_str = f"{ideal_min} {unit}+"
            if actual_val < ideal_min:
                penalty = (ideal_min - actual_val) * penalty_weight
                status = "needs_improvement"
        elif ideal_max is not None:
            ideal_str = f"≤{ideal_max} {unit}"
            if actual_val > ideal_max:
                penalty = (actual_val - ideal_max) * penalty_weight
                status = "needs_improvement"
        else:
            ideal_str = "N/A"

        penalty = min(penalty, 15.0) # 항목당 최대 감점 15점
        
        return {
            "name": name,
            "description": description,
            "actual": round(actual_val, 1) if isinstance(actual_val, float) else actual_val,
            "ideal": ideal_str,
            "status": status,
            "deduction": round(penalty, 1)
        }

    def _get_ref_length(self, pose_data, frame):
        """몸통 길이(어깨 중심~골반 중심) 계산 — 비례 기준"""
        KP = self.KP
        ref = 100.0
        if frame in pose_data:
            try:
                pts = pose_data[frame]
                ls, rs = pts[KP['left_shoulder']], pts[KP['right_shoulder']]
                lh, rh = pts[KP['left_hip']], pts[KP['right_hip']]
                mid_s = np.array([(ls[0]+rs[0])/2, (ls[1]+rs[1])/2])
                mid_h = np.array([(lh[0]+rh[0])/2, (lh[1]+rh[1])/2])
                ref = max(50.0, np.linalg.norm(mid_s - mid_h))
            except: pass
        return ref

    def calculate_score(self, pose_data, phases, dominant_hand='right'):
        """투핸드 프로 볼링 폼 기준 채점 (Belmonte/Palermaa 기반)"""
        evaluations = {k: [] for k in ["stance", "pushaway", "backswing", "forward_swing", "release"]}
        KP = self.KP
        
        s_f = phases.get('stance')
        pa_f = phases.get('pushaway_start')
        bs_start_f = phases.get('backswing_start')  # valley (pushaway 끝)
        bs_f = phases.get('backswing_top')
        rel_f = phases.get('release')
        ft_f = phases.get('follow_through_end')
        
        ref_length = self._get_ref_length(pose_data, s_f)
        non_dominant = 'left' if dominant_hand == 'right' else 'right'
        slide_side = non_dominant  # 슬라이딩 발은 비투구 쪽
        
        # ═══════════════════════════════════════════
        # 1. STANCE (준비 자세)
        # ═══════════════════════════════════════════
        if s_f in pose_data:
            s_pts = pose_data[s_f]
            # 양쪽 무릎 굽힘: 투핸드 프로는 살짝 낮은 자세로 시작 (155~170°)
            for side in ['left', 'right']:
                hip, knee, ankle = s_pts[KP[f'{side}_hip']], s_pts[KP[f'{side}_knee']], s_pts[KP[f'{side}_ankle']]
                evaluations["stance"].append(self._evaluate_metric(
                    f"Knee Bend ({side[0].upper()})", "Low center for stable start",
                    self.calculate_angle(hip, knee, ankle), 155, 175, 0.8, "°"))

        # ═══════════════════════════════════════════
        # 2. PUSHAWAY (푸시어웨이)
        # ═══════════════════════════════════════════
        if bs_start_f in pose_data and pa_f in pose_data:
            # 볼 드롭 높이: 푸시어웨이 시작 → 끝(valley)까지 손목이 얼마나 내려가는지
            pa_pts = pose_data[pa_f]
            valley_pts = pose_data[bs_start_f]
            wrist_start_y = pa_pts[KP[f'{dominant_hand}_wrist']][1]
            wrist_end_y = valley_pts[KP[f'{dominant_hand}_wrist']][1]
            drop_ratio = ((wrist_end_y - wrist_start_y) / ref_length) * 100
            evaluations["pushaway"].append(self._evaluate_metric(
                "Ball Drop", "Natural ball drop (vs torso)",
                drop_ratio, 20, 60, 0.8, "%"))

        # ═══════════════════════════════════════════
        # 3. BACKSWING (백스윙)
        # ═══════════════════════════════════════════
        if bs_f in pose_data:
            bs_pts = pose_data[bs_f]
            
            # 투구팔 팔꿈치 굽힘: 투핸드는 팔꿈치를 의도적으로 구부림 (90~150°)
            sh = bs_pts[KP[f'{dominant_hand}_shoulder']]
            el = bs_pts[KP[f'{dominant_hand}_elbow']]
            wr = bs_pts[KP[f'{dominant_hand}_wrist']]
            evaluations["backswing"].append(self._evaluate_metric(
                "Elbow Bend", "Two-hand ball control",
                self.calculate_angle(sh, el, wr), 90, 150, 1.0, "°"))
            
            # 척추 틸트: Belmonte ~60°, Palermaa ~90° → 범위 40~70°
            lh, rh = bs_pts[KP['left_hip']], bs_pts[KP['right_hip']]
            ls, rs = bs_pts[KP['left_shoulder']], bs_pts[KP['right_shoulder']]
            mid_h = [(lh[0]+rh[0])/2, (lh[1]+rh[1])/2]
            mid_s = [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2]
            tilt = 90 - np.degrees(np.arctan2(mid_h[1]-mid_s[1], abs(mid_s[0]-mid_h[0])))
            evaluations["backswing"].append(self._evaluate_metric(
                "Spine Tilt", "Upper body lean for power",
                tilt, 40, 70, 1.0, "°"))
            
            
            # Backswing Height: wrist vs HIP (two-hand: torso leans, ball above hip = strong)
            hip_y = (lh[1] + rh[1]) / 2
            wrist_y = wr[1]
            height_diff = ((hip_y - wrist_y) / ref_length) * 100  # + means wrist above hip
            evaluations["backswing"].append(self._evaluate_metric(
                "Backswing Height", "Wrist height vs hip",
                height_diff, 0, 80, 0.3, "%"))

        # ═══════════════════════════════════════════
        # 4. FORWARD SWING (포워드 스윙) — 기존에 비어있던 Phase
        # ═══════════════════════════════════════════
        if bs_f in pose_data and rel_f in pose_data:
            # [NEW] 다운스윙 팔꿈치 유지: 포워드 스윙 중간 지점에서 팔꿈치 굽힘 유지
            mid_frame = (bs_f + rel_f) // 2
            if mid_frame in pose_data:
                mid_pts = pose_data[mid_frame]
                sh = mid_pts[KP[f'{dominant_hand}_shoulder']]
                el = mid_pts[KP[f'{dominant_hand}_elbow']]
                wr = mid_pts[KP[f'{dominant_hand}_wrist']]
                evaluations["forward_swing"].append(self._evaluate_metric(
                    "Downswing Elbow", "Elbow tension in swing",
                    self.calculate_angle(sh, el, wr), 100, 160, 1.0, "°"))
            
            # [NEW] 무게 중심 이동: 백스윙 → 릴리즈 사이의 힙 X 이동량
            bs_pts = pose_data[bs_f]
            rel_pts = pose_data[rel_f]
            bs_hip_x = (bs_pts[KP['left_hip']][0] + bs_pts[KP['right_hip']][0]) / 2
            rel_hip_x = (rel_pts[KP['left_hip']][0] + rel_pts[KP['right_hip']][0]) / 2
            shift_ratio = (abs(rel_hip_x - bs_hip_x) / ref_length) * 100
            evaluations["forward_swing"].append(self._evaluate_metric(
                "Weight Shift", "Forward weight transfer",
                shift_ratio, 15, 60, 0.8, "%"))

        # ═══════════════════════════════════════════
        # 5. RELEASE (릴리즈 & 팔로우 스루)
        # ═══════════════════════════════════════════
        if rel_f in pose_data:
            rel_pts = pose_data[rel_f]
            
            # 슬라이딩 무릎 굽힘: Belmonte ~110°, 프로 기준 90~140°
            hip = rel_pts[KP[f'{slide_side}_hip']]
            knee = rel_pts[KP[f'{slide_side}_knee']]
            ankle = rel_pts[KP[f'{slide_side}_ankle']]
            evaluations["release"].append(self._evaluate_metric(
                "Slide Knee", "Low center of gravity",
                self.calculate_angle(hip, knee, ankle), 90, 140, 1.0, "°"))
            
            # 릴리즈 척추 틸트: 프로는 30~60° (Belmonte ~45°)
            lh, rh = rel_pts[KP['left_hip']], rel_pts[KP['right_hip']]
            ls, rs = rel_pts[KP['left_shoulder']], rel_pts[KP['right_shoulder']]
            mid_h = [(lh[0]+rh[0])/2, (lh[1]+rh[1])/2]
            mid_s = [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2]
            tilt = 90 - np.degrees(np.arctan2(mid_h[1]-mid_s[1], abs(mid_s[0]-mid_h[0])))
            evaluations["release"].append(self._evaluate_metric(
                "Spine Tilt", "Power transfer & head down",
                tilt, 30, 60, 1.5, "°"))
            
            # 머리 고정 (백스윙 시작 → 팔로우 스루 전체 구간)
            head_positions = []
            measure_start = bs_start_f if bs_start_f else (bs_f if bs_f else rel_f)
            measure_end = ft_f if ft_f else rel_f
            for f in range(measure_start, measure_end + 1):
                if f in pose_data:
                    kpts = pose_data[f]
                    n = kpts[KP['nose']]
                    le = kpts[KP['left_ear']]
                    re = kpts[KP['right_ear']]
                    avg_x = (n[0] + le[0] + re[0]) / 3
                    avg_y = (n[1] + le[1] + re[1]) / 3
                    head_positions.append((avg_x, avg_y))
                    
            if len(head_positions) > 5:
                jitter_px = np.std([p[1] for p in head_positions]) + np.std([p[0] for p in head_positions])
                jitter_ratio = (jitter_px / ref_length) * 100
                evaluations["release"].append(self._evaluate_metric(
                    "Head Stability", "Head movement in swing",
                    jitter_ratio, 0, 10, 1.0, "%"))

        total_deduction = sum(m.get("deduction", 0.0) for phase in evaluations.values() for m in phase)
        return {
            "overall_score": max(0.0, round(100.0 - total_deduction, 1)),
            "max_score": 100.0,
            "total_deduction": round(total_deduction, 1),
            "step_evaluations": evaluations
        }
    
    def calculate_summary_stats(self, pose_data, phases, fps):
        """사용자에게 보여줄 종합 통계 계산"""
        stats = {}
        KP = self.KP
        frames = sorted(pose_data.keys())
        
        pa_f = phases.get('pushaway_start', 0)
        ft_f = phases.get('follow_through_end', 0)
        bs_f = phases.get('backswing_top', 0)
        rel_f = phases.get('release', 0)
        s_f = phases.get('stance', 0)
        
        # 1. 전체 접근 시간 (푸시어웨이 → 팔로우 스루)
        if pa_f and ft_f and fps > 0:
            approach_time = (ft_f - pa_f) / fps
            stats['Approach Time'] = f"{approach_time:.2f}s"
        
        # 2. 백스윙 높이 (어깨 대비)
        if bs_f in pose_data:
            pts = pose_data[bs_f]
            ref_length = self._get_ref_length(pose_data, s_f)
            sh_y = pts[KP[f'right_shoulder']][1]
            wr_y = pts[KP[f'right_wrist']][1]
            height_pct = ((sh_y - wr_y) / ref_length) * 100
        # 2. Backswing Height (relative to hip - for two-handed form)
        if bs_f in pose_data:
            pts = pose_data[bs_f]
            ref_length = self._get_ref_length(pose_data, s_f)
            # Use hip Y as reference for two-handed backswing height
            hip_y = (pts[KP['left_hip']][1] + pts[KP['right_hip']][1]) / 2
            wr_y = pts[KP[f'right_wrist']][1]
            height_pct = ((hip_y - wr_y) / ref_length) * 100
            stats['Backswing Height'] = f"{height_pct:.0f}% of hip"
        
        # 3. Release Timing (Backswing top → Release)
        if bs_f and rel_f and fps > 0:
            release_time = (rel_f - bs_f) / fps
            stats['Release Timing'] = f"{release_time:.2f}s"
        
        # 5. 전체 동작 시간
        if len(frames) > 0 and fps > 0:
            total_time = (frames[-1] - frames[0]) / fps
            stats['Total Duration'] = f"{total_time:.2f}s"
        
        return stats

    def render_output(self, video_path, pose_data, phases, scores, output_path, summary_stats=None):
        """키포인트, 단계 오버레이, 점수 패널을 비디오에 렌더링"""
        print("Rendering video...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        orig_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Fixed output resolution
        target_width = 1080
        target_height = 1920
        
        # Calculate aspect ratios and scaling
        orig_aspect = orig_width / orig_height
        target_aspect = target_width / target_height
        
        if orig_aspect > target_aspect:
            # Video is wider than target: fit to width
            scale = target_width / orig_width
            new_w = target_width
            new_h = int(orig_height * scale)
            offset_x = 0
            offset_y = (target_height - new_h) // 2
        else:
            # Video is taller than target or same: fit to height
            scale = target_height / orig_height
            new_h = target_height
            new_w = int(orig_width * scale)
            offset_y = 0
            offset_x = (target_width - new_w) // 2

        temp_name = 'temp_rtmpose_no_audio.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_name, fourcc, fps, (target_width, target_height))
        
        delay_frames = int(0.15 * fps)
        
        try:
            # Reverted to original font sizes
            font = ImageFont.truetype(self.font_path, 52)
            small_font = ImageFont.truetype(self.font_path, 34)
            tiny_font = ImageFont.truetype(self.font_path, 26)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
            tiny_font = ImageFont.load_default()

        # === 🎨 Fun Theme 컬러 팔레트 ===
        # 각 Phase별 고유 컬러 (파스텔 + 비비드 혼합)
        phase_colors = {
            "stance":        (255, 179, 71, 255),   # 🍊 Warm Orange
            "pushaway":      (255, 105, 180, 255),   # 💗 Hot Pink
            "backswing":     (100, 149, 237, 255),   # 💎 Cornflower Blue
            "forward_swing": (50, 205, 50, 255),     # 🍀 Lime Green
            "release":       (255, 215, 0, 255),     # ⭐ Gold
        }

        # 스켈레톤 컬러 (트렌디한 시안/마젠타 그라데이션)
        skeleton_color_line = (0, 224, 208)    # Teal/Cyan
        skeleton_color_joint = (255, 0, 128)   # Magenta-ish
        head_color = (255, 220, 50)            # Bright Yellow

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Create black canvas
            canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            # Scale video frame
            resized_frame = cv2.resize(frame, (new_w, new_h))
            # Place scaled frame on canvas
            canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized_frame

            if frame_count in pose_data:
                kpts = pose_data[frame_count]
                # RTMPose 스켈레톤 - applying scale and offset
                links = [
                    (5, 7), (7, 9),     # Left arm
                    (6, 8), (8, 10),    # Right arm
                    (11, 13), (13, 15), # Left leg
                    (12, 14), (14, 16), # Right leg
                    (5, 6), (11, 12), (5, 11), (6, 12)  # Torso
                ]
                for (i, j) in links:
                    pt1 = (int(kpts[i][0] * scale + offset_x), int(kpts[i][1] * scale + offset_y))
                    pt2 = (int(kpts[j][0] * scale + offset_x), int(kpts[j][1] * scale + offset_y))
                    cv2.line(canvas, pt1, pt2, skeleton_color_line, 3)
                    
                # 머리 중심점
                head_x = sum(kpts[i][0] for i in range(5)) / 5
                head_y = sum(kpts[i][1] for i in range(5)) / 5
                head_pt = (int(head_x * scale + offset_x), int(head_y * scale + offset_y))
                
                mid_shoulder_x = (kpts[5][0] + kpts[6][0]) / 2
                mid_shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
                mid_shoulder = (int(mid_shoulder_x * scale + offset_x), int(mid_shoulder_y * scale + offset_y))
                cv2.line(canvas, mid_shoulder, head_pt, skeleton_color_line, 3)
                cv2.circle(canvas, head_pt, 8, head_color, -1)
                cv2.circle(canvas, head_pt, 10, head_color, 2)

                # 관절 포인트 (Magenta)
                for i in range(5, 17):
                    pt = (int(kpts[i][0] * scale + offset_x), int(kpts[i][1] * scale + offset_y))
                    cv2.circle(canvas, pt, 5, skeleton_color_joint, -1)
                    cv2.circle(canvas, pt, 7, (255, 255, 255), 1)  # White outline

            # PIL Overlay
            img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).convert("RGBA")
            overlay = Image.new('RGBA', img_pil.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Phase 판별 (Delay 적용)
            current_phase_name = "1. Stance"
            current_phase_key = "stance"
            
            if frame_count >= phases.get('release', 0) + delay_frames:
                current_phase_name = "5. Release & Follow-through"
                current_phase_key = "release"
            elif frame_count >= phases.get('backswing_top', 0) + delay_frames:
                current_phase_name = "4. Forward Swing"
                current_phase_key = "forward_swing"
            elif frame_count >= phases.get('backswing_start', 0) + delay_frames:
                current_phase_name = "3. Back Swing"
                current_phase_key = "backswing"
            elif frame_count >= phases.get('pushaway_start', 0) + delay_frames:
                current_phase_name = "2. Push-away"
                current_phase_key = "pushaway"

            # ╔════════════════════════════════════════╗
            # ║   🎨 FUN THEME: 상단 Phase Tab Bar     ║
            # ╚════════════════════════════════════════╝
            phases_list = [
                ("Stance", "stance"),
                ("Push", "pushaway"),
                ("Back", "backswing"),
                ("Fwd", "forward_swing"),
                ("Release", "release")
            ]
            
            bar_height = 75
            # 상단 바 배경 (깔끔한 다크)
            draw.rectangle([0, 0, target_width, bar_height], fill=(18, 18, 32, 255))
            # 상단 구분선 (얇은 그라데이션 느낌)
            draw.line([0, bar_height - 1, target_width, bar_height - 1], fill=(60, 60, 80, 255), width=1)
            
            segment_width = target_width / len(phases_list)
            
            for idx, (phase_title, phase_key) in enumerate(phases_list):
                x0 = idx * segment_width
                x1 = (idx + 1) * segment_width
                this_color = phase_colors[phase_key]
                
                label = phase_title
                bbox = draw.textbbox((0, 0), label, font=small_font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                
                # 텍스트 위치를 바 전체 기준으로 고정 (active/inactive 동일한 Y)
                center_x = x0 + (segment_width - text_w) / 2
                center_y = (bar_height - text_h) / 2 - bbox[1]
                
                if phase_key == current_phase_key:
                    # 활성 탭: 배경 Pill 먼저 그리고 텍스트는 같은 Y에 배치
                    margin_x = 3
                    margin_y = 8
                    draw.rounded_rectangle(
                        [x0 + margin_x, margin_y, x1 - margin_x, bar_height - margin_y],
                        radius=18, fill=this_color
                    )
                    draw.text((center_x, center_y), label, font=small_font, fill=(20, 20, 20, 255))
                    
                    # 하단 도트 인디케이터
                    dot_x = x0 + segment_width / 2
                    dot_y = bar_height - 5
                    draw.ellipse([dot_x - 4, dot_y - 4, dot_x + 4, dot_y + 4], fill=(255, 255, 255, 255))
                else:
                    # 비활성 탭: 희미한 텍스트 (같은 Y 위치)
                    faded_color = (this_color[0], this_color[1], this_color[2], 120)
                    draw.text((center_x, center_y), label, font=small_font, fill=faded_color)

            # ╔════════════════════════════════════════╗
            # ║   📊 FUN THEME: 메트릭 카드 패널       ║
            # ╚════════════════════════════════════════╝
            current_metrics = scores['step_evaluations'].get(current_phase_key, [])
            active_color = phase_colors[current_phase_key]
            
            if current_metrics:
                padding = 28
                line_height = 62
                drop_y0 = bar_height
                drop_y1 = drop_y0 + padding * 2 + len(current_metrics) * line_height
                
                # 메트릭 패널 배경 (살짝 밝은 다크)
                draw.rectangle([0, drop_y0, target_width, drop_y1], fill=(24, 24, 42, 255))
                # 현재 Phase 컬러로 얇은 상단 액센트 라인
                draw.rectangle([0, drop_y0, target_width, drop_y0 + 3], fill=active_color)
                
                # 상태 아이콘 & 색상
                metric_y = drop_y0 + padding
                for metric in current_metrics:
                    is_good = metric['status'] == 'perfect'
                    dot_color = (80, 255, 120, 255) if is_good else (255, 100, 80, 255)
                    val_color = dot_color
                    
                    # 상태 도트 (컬러 원)
                    left_margin = 30
                    draw.ellipse([left_margin, metric_y + 10, left_margin + 16, metric_y + 26], fill=dot_color)
                    
                    # 항목 이름 (흰색)
                    name_x = left_margin + 50
                    draw.text((name_x, metric_y), str(metric['name']), font=small_font, fill=(255, 255, 255, 255))
                    
                    # 값 (오른쪽 정렬, 색상으로 상태 표시)
                    val_text = f"{metric['actual']}"
                    bbox_val = draw.textbbox((0, 0), val_text, font=small_font)
                    val_w = bbox_val[2] - bbox_val[0]
                    val_x = target_width - val_w - 30
                    draw.text((val_x, metric_y), val_text, font=small_font, fill=val_color)
                    
                    # 기준값 (작은 글씨로 값 밑에)
                    ideal_text = f"({metric['ideal']})"
                    bbox_ideal = draw.textbbox((0, 0), ideal_text, font=tiny_font)
                    ideal_w = bbox_ideal[2] - bbox_ideal[0]
                    draw.text((val_x + (val_w - ideal_w) / 2, metric_y + 30), ideal_text, font=tiny_font, fill=(130, 130, 160, 255))
                    
                    metric_y += line_height

            # ╔════════════════════════════════════════╗
            # ║   🏆 FUN THEME: 하단 점수 배지         ║
            # ╚════════════════════════════════════════╝
            score_val = scores['overall_score']
            if score_val >= 80:
                neon_color = (0, 255, 140)
            elif score_val >= 60:
                neon_color = (60, 180, 255)
            elif score_val >= 40:
                neon_color = (255, 180, 50)
            else:
                neon_color = (255, 80, 80)
            
            score_text = f"{score_val}"
            sub_text = "/ 100"
            bbox = draw.textbbox((0, 0), score_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            bbox_sub = draw.textbbox((0, 0), sub_text, font=small_font)
            sub_w = bbox_sub[2] - bbox_sub[0]
            
            total_w = text_w + sub_w + 12
            badge_w = total_w + 70
            badge_h = text_h + 40
            badge_x = (target_width - badge_w) / 2
            badge_y = target_height - badge_h - 120
            
            # Neon glow score badge
            for glow in range(3, 0, -1):
                g_expand = glow * 5
                g_alpha = 20 + (3 - glow) * 15
                # Outer glow layers (expanding neon rings)
                draw.rounded_rectangle(
                    [badge_x - g_expand, badge_y - g_expand,
                     badge_x + badge_w + g_expand, badge_y + badge_h + g_expand],
                    radius=(badge_h + g_expand * 2) // 2,
                    fill=(neon_color[0], neon_color[1], neon_color[2], g_alpha)
                )
            
            # Dark frosted backdrop
            draw.rounded_rectangle(
                [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
                radius=badge_h // 2, fill=(12, 12, 24, 220)
            )
            # Neon border
            draw.rounded_rectangle(
                [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
                radius=badge_h // 2, outline=(*neon_color, 200), width=3
            )
            
            # Score text (bright white)
            score_x = badge_x + (badge_w - total_w) / 2
            score_y = badge_y + (badge_h - text_h) / 2 - bbox[1]
            draw.text((score_x, score_y), score_text, font=font, fill=(255, 255, 255, 255))
            # "/ 100" in neon accent color, dimmed
            sub_y = badge_y + (badge_h - (bbox_sub[3] - bbox_sub[1])) / 2 - bbox_sub[1]
            draw.text((score_x + text_w + 12, sub_y), sub_text, font=small_font, fill=(*neon_color, 180))

            final_img = Image.alpha_composite(img_pil, overlay)
            out.write(cv2.cvtColor(np.array(final_img), cv2.COLOR_RGB2BGR))
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Rendering... {frame_count} frames")

        cap.release()
        
        # ╔════════════════════════════════════════════════╗
        # ║   END SUMMARY: 2초간 전체 데이터 요약 화면     ║
        # ╚════════════════════════════════════════════════╝
        
        summary_frames = int(fps * 2)
        
        # summary background (solid dark) at target resolution
        summary_bg = Image.new('RGBA', (target_width, target_height), (18, 18, 32, 255))
        summary_draw = ImageDraw.Draw(summary_bg)
        
        # --- Top: score displayed large ---
        score_val = scores['overall_score']
        
        try:
            # Reverted to original summary font sizes
            title_font = ImageFont.truetype(self.font_path, 64)
            section_font = ImageFont.truetype(self.font_path, 36)
            data_font = ImageFont.truetype(self.font_path, 30)
            sub_font = ImageFont.truetype(self.font_path, 22)
        except:
            title_font = font
            section_font = small_font
            data_font = small_font
            sub_font = tiny_font
        
        y_cursor = 60
        
        # Title: "BOWLING ANALYSIS"
        title_text = "BOWLING ANALYSIS"
        bbox = summary_draw.textbbox((0, 0), title_text, font=section_font)
        tw = bbox[2] - bbox[0]
        summary_draw.text(((target_width - tw) / 2, y_cursor), title_text, font=section_font, fill=(180, 180, 200, 255))
        y_cursor += 70
        
        # Neon glow score pill (matching video badge)
        if score_val >= 80:
            neon_s = (0, 255, 140)
        elif score_val >= 60:
            neon_s = (60, 180, 255)
        elif score_val >= 40:
            neon_s = (255, 180, 50)
        else:
            neon_s = (255, 80, 80)
        
        score_text = f"{score_val}"
        bbox = summary_draw.textbbox((0, 0), score_text, font=title_font)
        stw = bbox[2] - bbox[0]
        sth = bbox[3] - bbox[1]
        
        pill_w = stw + 100
        pill_h = sth + 40
        pill_x = (target_width - pill_w) / 2
        pill_y = y_cursor
        
        # Outer glow rings
        for glow in range(3, 0, -1):
            g_expand = glow * 6
            g_alpha = 15 + (3 - glow) * 12
            summary_draw.rounded_rectangle(
                [pill_x - g_expand, pill_y - g_expand,
                 pill_x + pill_w + g_expand, pill_y + pill_h + g_expand],
                radius=(pill_h + g_expand * 2) // 2,
                fill=(neon_s[0], neon_s[1], neon_s[2], g_alpha)
            )
        # Dark backdrop
        summary_draw.rounded_rectangle(
            [pill_x, pill_y, pill_x + pill_w, pill_y + pill_h],
            radius=pill_h // 2, fill=(12, 12, 24, 240)
        )
        # Neon border
        summary_draw.rounded_rectangle(
            [pill_x, pill_y, pill_x + pill_w, pill_y + pill_h],
            radius=pill_h // 2, outline=(*neon_s, 200), width=3
        )
        # Score text centered (white)
        score_text_x = pill_x + (pill_w - stw) / 2
        score_text_y = pill_y + (pill_h - sth) / 2 - bbox[1]
        summary_draw.text((score_text_x, score_text_y), score_text, font=title_font, fill=(255, 255, 255, 255))
        
        # "/100" suffix
        sub_score = "/ 100"
        bbox_sub = summary_draw.textbbox((0, 0), sub_score, font=section_font)
        summary_draw.text((pill_x + pill_w + 15, pill_y + pill_h / 2 - 10), sub_score, font=section_font, fill=(180, 180, 200, 255))
        
        y_cursor = pill_y + pill_h + 50
        
        # divider
        summary_draw.line([40, y_cursor, target_width - 40, y_cursor], fill=(60, 60, 80, 255), width=2)
        y_cursor += 35
        
        # --- Phase summaries ---
        phases_summary = [
            ("Stance", "stance"),
            ("Push-away", "pushaway"),
            ("Back Swing", "backswing"),
            ("Fwd Swing", "forward_swing"),
            ("Release", "release")
        ]
        
        for phase_title, phase_key in phases_summary:
            metrics = scores['step_evaluations'].get(phase_key, [])
            if not metrics:
                continue
            
            this_color = phase_colors[phase_key]
            
            # Phase header — sleek outline badge with accent line
            header_text = phase_title
            bbox_h = summary_draw.textbbox((0, 0), header_text, font=section_font)
            h_w = bbox_h[2] - bbox_h[0]
            h_h = bbox_h[3] - bbox_h[1]
            
            badge_pad_x = 16
            badge_pad_y = 6
            badge_left = 30
            badge_top = y_cursor
            badge_right = badge_left + h_w + badge_pad_x * 2
            badge_bottom = badge_top + h_h + badge_pad_y * 2
            
            # Dark fill with colored left accent line
            summary_draw.rounded_rectangle(
                [badge_left, badge_top, badge_right, badge_bottom],
                radius=8, fill=(30, 30, 50, 255), outline=(*this_color[:3], 120), width=1
            )
            # Colored accent bar on left
            summary_draw.rectangle(
                [badge_left, badge_top + 4, badge_left + 4, badge_bottom - 4],
                fill=this_color
            )
            header_text_y = badge_top + (badge_bottom - badge_top - h_h) / 2 - bbox_h[1]
            summary_draw.text((badge_left + badge_pad_x + 6, header_text_y), header_text, font=section_font, fill=this_color)
            y_cursor = badge_bottom + 15
            
            # Column headers — phase name replaces "Metric"
            col_name_x = 50
            col_val_x = target_width * 0.55
            col_ideal_x = target_width * 0.78
            
            summary_draw.text((col_name_x, y_cursor), "Metric", font=sub_font, fill=(100, 100, 130, 255))
            summary_draw.text((col_val_x, y_cursor), "Value", font=sub_font, fill=(100, 100, 130, 255))
            summary_draw.text((col_ideal_x, y_cursor), "Pro", font=sub_font, fill=(100, 100, 130, 255))
            y_cursor += 32
            
            # 얀은 구분선
            summary_draw.line([col_name_x, y_cursor, target_width - 30, y_cursor], fill=(50, 50, 70, 255), width=1)
            y_cursor += 12
            
            # 각 메트릭 행
            for metric in metrics:
                is_good = metric['status'] == 'perfect'
                dot_color = (80, 255, 120, 255) if is_good else (255, 100, 80, 255)
                val_color = dot_color
                
                # 상태 도트
                summary_draw.ellipse([col_name_x, y_cursor + 8, col_name_x + 14, y_cursor + 22], fill=dot_color)
                # 항목명
                name_text = f"{metric['name']}"
                summary_draw.text((col_name_x + 24, y_cursor), name_text, font=data_font, fill=(255, 255, 255, 255))
                # 실측값 (칼럼 정렬)
                val_text = f"{metric['actual']}"
                summary_draw.text((col_val_x, y_cursor), val_text, font=data_font, fill=val_color)
                # Pro 기준 (칼럼 정렬)
                ideal_text = f"{metric['ideal']}"
                summary_draw.text((col_ideal_x, y_cursor), ideal_text, font=data_font, fill=(150, 150, 180, 255))
                
                y_cursor += 46
            
            # Phase 간 여백
            y_cursor += 25
        
        # --- Total stats section ---
        if summary_stats:
            summary_draw.line([40, y_cursor + 5, target_width - 40, y_cursor + 5], fill=(60, 60, 80, 255), width=2)
            y_cursor += 30
            
            stat_title = "GENERAL STATS"
            bbox_st = summary_draw.textbbox((0, 0), stat_title, font=section_font)
            st_w = bbox_st[2] - bbox_st[0]
            st_h = bbox_st[3] - bbox_st[1]
            
            stat_badge_pad_x = 20
            stat_badge_pad_y = 10
            stat_badge_left = 30
            stat_badge_top = y_cursor
            stat_badge_right = stat_badge_left + st_w + stat_badge_pad_x * 2
            stat_badge_bottom = stat_badge_top + st_h + stat_badge_pad_y * 2
            
            summary_draw.rounded_rectangle(
                [stat_badge_left, stat_badge_top, stat_badge_right, stat_badge_bottom],
                radius=10, fill=(120, 120, 180, 255)
            )
            stat_text_y = stat_badge_top + (stat_badge_bottom - stat_badge_top - st_h) / 2 - bbox_st[1]
            summary_draw.text((stat_badge_left + stat_badge_pad_x, stat_text_y), stat_title, font=section_font, fill=(20, 20, 20, 255))
            y_cursor = stat_badge_bottom + 20
            
            for stat_name, stat_val in summary_stats.items():
                summary_draw.text((50, y_cursor), stat_name, font=data_font, fill=(200, 200, 220, 255))
                bbox_sv = summary_draw.textbbox((0, 0), str(stat_val), font=data_font)
                sv_w = bbox_sv[2] - bbox_sv[0]
                summary_draw.text((target_width - sv_w - 50, y_cursor), str(stat_val), font=data_font, fill=(130, 200, 255, 255))
                y_cursor += 42
        
        # 서머리 프레임을 BGR로 변환하여 2초간 반복 기록
        summary_bgr = cv2.cvtColor(np.array(summary_bg.convert('RGB')), cv2.COLOR_RGB2BGR)
        for _ in range(summary_frames):
            out.write(summary_bgr)
        
        print(f"Summary screen: {summary_frames} frames added")
        
        out.release()
        
        # Audio multiplexing
        try:
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            (
                ffmpeg
                .output(ffmpeg.input(temp_name), ffmpeg.input(video_path).audio, output_path, vcodec='libx264', acodec='aac', strict='experimental', loglevel='error')
                .overwrite_output()
                .run(cmd=ffmpeg_path)
            )
            os.remove(temp_name)
            print(f"Video saved: {output_path}")
        except Exception as e:
            print(f"Audio merge failed: {e}")

    def run(self, video_path):
        base_name = os.path.basename(video_path)
        name, _ = os.path.splitext(base_name)
        out_json_pts = f"rtmpose_{name}_keypoints.json"
        out_json_score = f"rtmpose_{name}_score.json"
        out_video = f"rtmpose_{name}_analyzed.mp4"

        pose_data = self.extract_keypoints(video_path)
        with open(out_json_pts, 'w', encoding='utf-8') as f:
            json.dump({str(k): [float(p) for p in pt] for k, pts in pose_data.items() for pt in pts}, f) # Simplify format for storage if needed, but keeping simple here
            
        phases = self.segment_phases(pose_data)
        
        # 보정: phase가 전혀 안 찾아지면 기본값 세팅
        if not phases:
            print("Could not detect phases. Aborting analysis.")
            return
            
        scores = self.calculate_score(pose_data, phases)
        
        # FPS 계산 (summary stats 용)
        cap_tmp = cv2.VideoCapture(video_path)
        fps = cap_tmp.get(cv2.CAP_PROP_FPS)
        cap_tmp.release()
        
        summary_stats = self.calculate_summary_stats(pose_data, phases, fps)
        
        with open(out_json_score, 'w', encoding='utf-8') as f:
            json.dump(scores, f, indent=4, ensure_ascii=False)
            
        print("\n=== Analysis Score ===")
        print(f"Total: {scores['overall_score']}")
            
        self.render_output(video_path, pose_data, phases, scores, out_video, summary_stats)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='/Users/meadows/Desktop/Coding/Antigravity Project/Bowling_Swing_Mechanism_Scorer/BV_3.mov', help='Input video file path')
    args = parser.parse_args()
    
    pipeline = RTMPoseBowlingPipeline()
    pipeline.run(args.video)

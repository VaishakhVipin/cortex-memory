#!/usr/bin/env python3
"""
🧠 Cortex Semantic Drift Detection
Monitors system performance and detects behavioral changes over time.
"""

import json
import time
import statistics
import math
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from collections import defaultdict
from .redis_client import r
from .semantic_embeddings import semantic_embeddings

class SemanticDriftDetection:
    """
    Detects semantic drift in the context model using statistical algorithms:
    - Performance degradation over time
    - User behavior pattern shifts
    - Context relevance changes
    - System accuracy drift
    """
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.drift_threshold = 0.15  # 15% performance drop triggers alert
        self.window_size = 30  # Days to analyze for drift
        self.min_data_points = 10  # Minimum data points for reliable detection
        self.drift_cache_ttl = 1800  # 30 minutes cache
        
        # Drift detection configuration
        self.performance_monitoring_enabled = True
        self.behavioral_drift_enabled = True
        self.context_relevance_drift_enabled = True
        self.accuracy_drift_enabled = True
    
    def detect_semantic_drift(self, user_id: str) -> Dict:
        """
        Comprehensive semantic drift detection using statistical algorithms.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with drift analysis results
        """
        r = self.redis_client
        
        drift_analysis = {
            'overall_drift_score': 0.0,
            'drift_detected': False,
            'drift_severity': 'none',
            'drift_components': {},
            'performance_drift': {},
            'behavioral_drift': {},
            'context_relevance_drift': {},
            'accuracy_drift': {},
            'statistical_insights': {},
            'recommendations': [],
            'trends': {},
            'alerts': []
        }
        
        # Check cache first
        cache_key = f"drift_analysis:{user_id}"
        cached_data = r.get(cache_key)
        if cached_data:
            try:
                drift_analysis = json.loads(cached_data)
                print(f"📊 Using cached drift analysis for user: {user_id}")
                return drift_analysis
            except json.JSONDecodeError:
                pass
        
        try:
            # Get historical data
            historical_data = self._get_historical_data(user_id)
            
            if not historical_data or len(historical_data.get('performance_metrics', [])) < self.min_data_points:
                drift_analysis['alerts'].append("Insufficient data for drift detection")
                return drift_analysis
            
            # Detect different types of drift
            if self.performance_monitoring_enabled:
                drift_analysis['performance_drift'] = self._detect_performance_drift(user_id, historical_data)
            
            if self.behavioral_drift_enabled:
                drift_analysis['behavioral_drift'] = self._detect_behavioral_drift(user_id, historical_data)
            
            if self.context_relevance_drift_enabled:
                drift_analysis['context_relevance_drift'] = self._detect_context_relevance_drift(user_id, historical_data)
            
            if self.accuracy_drift_enabled:
                drift_analysis['accuracy_drift'] = self._detect_accuracy_drift(user_id, historical_data)
            
            # Calculate overall drift score
            drift_analysis['overall_drift_score'] = self._calculate_overall_drift_score(drift_analysis)
            drift_analysis['drift_detected'] = drift_analysis['overall_drift_score'] > self.drift_threshold
            drift_analysis['drift_severity'] = self._determine_drift_severity(drift_analysis['overall_drift_score'])
            
            # Generate insights and recommendations
            drift_analysis['statistical_insights'] = self._get_statistical_insights(user_id, historical_data)
            drift_analysis['recommendations'] = self._generate_drift_recommendations(drift_analysis)
            drift_analysis['trends'] = self._analyze_drift_trends(historical_data)
            drift_analysis['alerts'] = self._generate_drift_alerts(drift_analysis)
            
            # Cache results
            r.setex(cache_key, self.drift_cache_ttl, json.dumps(drift_analysis))
            
            print(f"🔍 Drift analysis completed for user: {user_id}")
            return drift_analysis
            
        except Exception as e:
            print(f"❌ Error in drift detection: {e}")
            drift_analysis['alerts'].append(f"Drift detection error: {str(e)}")
            return drift_analysis
    
    def _get_statistical_insights(self, user_id: str, historical_data: Dict) -> Dict:
        """Generate statistical insights from historical data."""
        insights = {
            'data_quality': {},
            'trend_analysis': {},
            'anomaly_detection': {},
            'correlation_analysis': {}
        }
        
        try:
            performance_metrics = historical_data.get('performance_metrics', [])
            if len(performance_metrics) >= 5:
                # Data quality analysis
                response_times = [m.get('response_time', 0) for m in performance_metrics]
                success_rates = [m.get('success_rate', 0) for m in performance_metrics]
                
                insights['data_quality'] = {
                    'total_data_points': len(performance_metrics),
                    'avg_response_time': statistics.mean(response_times),
                    'response_time_std': statistics.stdev(response_times) if len(response_times) > 1 else 0,
                    'avg_success_rate': statistics.mean(success_rates),
                    'success_rate_std': statistics.stdev(success_rates) if len(success_rates) > 1 else 0
                }
                
                # Trend analysis
                if len(response_times) >= 7:
                    recent_avg = statistics.mean(response_times[-7:])
                    older_avg = statistics.mean(response_times[:-7])
                    trend = "improving" if recent_avg < older_avg else "degrading"
                    
                    insights['trend_analysis'] = {
                        'performance_trend': trend,
                        'trend_magnitude': abs(recent_avg - older_avg) / older_avg if older_avg > 0 else 0
                    }
                
                # Simple anomaly detection using z-score
                if len(response_times) > 1:
                    mean_rt = statistics.mean(response_times)
                    std_rt = statistics.stdev(response_times)
                    anomalies = [i for i, rt in enumerate(response_times) if abs(rt - mean_rt) > 2 * std_rt]
                    
                    insights['anomaly_detection'] = {
                        'anomaly_count': len(anomalies),
                        'anomaly_indices': anomalies,
                        'anomaly_threshold': mean_rt + 2 * std_rt
                    }
        
        except Exception as e:
            print(f"⚠️ Error generating statistical insights: {e}")
        
        return insights
    
    def _get_historical_data(self, user_id: str) -> Dict:
        """Get historical data for drift analysis."""
        r = self.redis_client
        
        historical_data = {
            'performance_metrics': [],
            'behavioral_patterns': [],
            'context_usage': [],
            'accuracy_metrics': []
        }
        
        try:
            # Get performance metrics
            performance_keys = r.keys(f"performance:{user_id}:*")
            for key in performance_keys[-self.window_size:]:  # Last 30 days
                data = r.get(key)
                if data:
                    try:
                        metric_data = json.loads(data)
                        historical_data['performance_metrics'].append(metric_data)
                    except json.JSONDecodeError:
                        continue
            
            # Get behavioral patterns
            behavior_keys = r.keys(f"behavior:{user_id}:*")
            for key in behavior_keys[-self.window_size:]:
                data = r.get(key)
                if data:
                    try:
                        behavior_data = json.loads(data)
                        historical_data['behavioral_patterns'].append(behavior_data)
                    except json.JSONDecodeError:
                        continue
            
            # Get context usage data
            context_keys = r.keys(f"context_usage:{user_id}:*")
            for key in context_keys[-self.window_size:]:
                data = r.get(key)
                if data:
                    try:
                        context_data = json.loads(data)
                        historical_data['context_usage'].append(context_data)
                    except json.JSONDecodeError:
                        continue
            
            # Get accuracy metrics
            accuracy_keys = r.keys(f"accuracy:{user_id}:*")
            for key in accuracy_keys[-self.window_size:]:
                data = r.get(key)
                if data:
                    try:
                        accuracy_data = json.loads(data)
                        historical_data['accuracy_metrics'].append(accuracy_data)
                    except json.JSONDecodeError:
                        continue
        
        except Exception as e:
            print(f"⚠️ Error getting historical data: {e}")
        
        return historical_data
    
    def _detect_performance_drift(self, user_id: str, historical_data: Dict) -> Dict:
        """Detect performance degradation over time."""
        drift_info = {
            'detected': False,
            'score': 0.0,
            'trend': 'stable',
            'metrics': {}
        }
        
        try:
            performance_metrics = historical_data.get('performance_metrics', [])
            if len(performance_metrics) < 5:
                return drift_info
            
            # Analyze response times
            response_times = [m.get('response_time', 0) for m in performance_metrics]
            if len(response_times) >= 7:
                recent_avg = statistics.mean(response_times[-7:])
                older_avg = statistics.mean(response_times[:-7])
                
                if older_avg > 0:
                    performance_change = (recent_avg - older_avg) / older_avg
                    drift_info['score'] = max(0, performance_change)  # Only positive changes indicate degradation
                    drift_info['detected'] = performance_change > 0.1  # 10% degradation threshold
                    drift_info['trend'] = 'degrading' if performance_change > 0.05 else 'stable'
                    drift_info['metrics']['response_time_change'] = performance_change
            
            # Analyze success rates
            success_rates = [m.get('success_rate', 1.0) for m in performance_metrics]
            if len(success_rates) >= 7:
                recent_success = statistics.mean(success_rates[-7:])
                older_success = statistics.mean(success_rates[:-7])
                
                success_change = older_success - recent_success  # Decrease in success rate
                drift_info['metrics']['success_rate_change'] = success_change
                
                if success_change > 0.05:  # 5% drop in success rate
                    drift_info['detected'] = True
                    drift_info['score'] = max(drift_info['score'], success_change)
        
        except Exception as e:
            print(f"⚠️ Error in performance drift detection: {e}")
        
        return drift_info
    
    def _detect_behavioral_drift(self, user_id: str, historical_data: Dict) -> Dict:
        """Detect changes in user behavior patterns."""
        drift_info = {
            'detected': False,
            'score': 0.0,
            'pattern_changes': []
        }
        
        try:
            behavioral_patterns = historical_data.get('behavioral_patterns', [])
            if len(behavioral_patterns) < 5:
                return drift_info
            
            # Analyze query complexity changes
            complexities = [p.get('query_complexity', 0) for p in behavioral_patterns]
            if len(complexities) >= 7:
                recent_complexity = statistics.mean(complexities[-7:])
                older_complexity = statistics.mean(complexities[:-7])
                
                complexity_change = abs(recent_complexity - older_complexity) / max(older_complexity, 1)
                if complexity_change > 0.2:  # 20% change in complexity
                    drift_info['detected'] = True
                    drift_info['score'] = complexity_change
                    drift_info['pattern_changes'].append(f"Query complexity changed by {complexity_change:.1%}")
            
            # Analyze topic distribution changes
            topics = [p.get('primary_topic', 'general') for p in behavioral_patterns]
            if len(topics) >= 10:
                recent_topics = topics[-10:]
                older_topics = topics[:-10]
                
                recent_dist = defaultdict(int)
                older_dist = defaultdict(int)
                
                for topic in recent_topics:
                    recent_dist[topic] += 1
                for topic in older_topics:
                    older_dist[topic] += 1
                
                # Calculate distribution similarity
                all_topics = set(recent_dist.keys()) | set(older_dist.keys())
                total_diff = 0
                
                for topic in all_topics:
                    recent_pct = recent_dist[topic] / len(recent_topics)
                    older_pct = older_dist[topic] / len(older_topics)
                    total_diff += abs(recent_pct - older_pct)
                
                if total_diff > 0.3:  # 30% change in topic distribution
                    drift_info['detected'] = True
                    drift_info['score'] = max(drift_info['score'], total_diff)
                    drift_info['pattern_changes'].append(f"Topic distribution changed by {total_diff:.1%}")
        
        except Exception as e:
            print(f"⚠️ Error in behavioral drift detection: {e}")
        
        return drift_info
    
    def _detect_context_relevance_drift(self, user_id: str, historical_data: Dict) -> Dict:
        """Detect changes in context relevance over time."""
        drift_info = {
            'detected': False,
            'score': 0.0,
            'relevance_metrics': {}
        }
        
        try:
            context_usage = historical_data.get('context_usage', [])
            if len(context_usage) < 5:
                return drift_info
            
            # Analyze context hit rates
            hit_rates = [c.get('context_hit_rate', 0) for c in context_usage]
            if len(hit_rates) >= 7:
                recent_hit_rate = statistics.mean(hit_rates[-7:])
                older_hit_rate = statistics.mean(hit_rates[:-7])
                
                hit_rate_change = older_hit_rate - recent_hit_rate  # Decrease in hit rate
                drift_info['relevance_metrics']['hit_rate_change'] = hit_rate_change
                
                if hit_rate_change > 0.1:  # 10% drop in hit rate
                    drift_info['detected'] = True
                    drift_info['score'] = hit_rate_change
            
            # Analyze context similarity scores
            similarity_scores = [c.get('avg_similarity', 0) for c in context_usage]
            if len(similarity_scores) >= 7:
                recent_similarity = statistics.mean(similarity_scores[-7:])
                older_similarity = statistics.mean(similarity_scores[:-7])
                
                similarity_change = older_similarity - recent_similarity  # Decrease in similarity
                drift_info['relevance_metrics']['similarity_change'] = similarity_change
                
                if similarity_change > 0.1:  # 10% drop in similarity
                    drift_info['detected'] = True
                    drift_info['score'] = max(drift_info['score'], similarity_change)
        
        except Exception as e:
            print(f"⚠️ Error in context relevance drift detection: {e}")
        
        return drift_info
    
    def _detect_accuracy_drift(self, user_id: str, historical_data: Dict) -> Dict:
        """Detect changes in system accuracy over time."""
        drift_info = {
            'detected': False,
            'score': 0.0,
            'accuracy_metrics': {}
        }
        
        try:
            accuracy_metrics = historical_data.get('accuracy_metrics', [])
            if len(accuracy_metrics) < 5:
                return drift_info
            
            # Analyze user satisfaction scores
            satisfaction_scores = [a.get('user_satisfaction', 0) for a in accuracy_metrics]
            if len(satisfaction_scores) >= 7:
                recent_satisfaction = statistics.mean(satisfaction_scores[-7:])
                older_satisfaction = statistics.mean(satisfaction_scores[:-7])
                
                satisfaction_change = older_satisfaction - recent_satisfaction  # Decrease in satisfaction
                drift_info['accuracy_metrics']['satisfaction_change'] = satisfaction_change
                
                if satisfaction_change > 0.1:  # 10% drop in satisfaction
                    drift_info['detected'] = True
                    drift_info['score'] = satisfaction_change
            
            # Analyze response quality scores
            quality_scores = [a.get('response_quality', 0) for a in accuracy_metrics]
            if len(quality_scores) >= 7:
                recent_quality = statistics.mean(quality_scores[-7:])
                older_quality = statistics.mean(quality_scores[:-7])
                
                quality_change = older_quality - recent_quality  # Decrease in quality
                drift_info['accuracy_metrics']['quality_change'] = quality_change
                
                if quality_change > 0.1:  # 10% drop in quality
                    drift_info['detected'] = True
                    drift_info['score'] = max(drift_info['score'], quality_change)
        
        except Exception as e:
            print(f"⚠️ Error in accuracy drift detection: {e}")
        
        return drift_info
    
    def _calculate_trend(self, values: List[float], days: List[float]) -> str:
        """Calculate trend direction using linear regression."""
        if len(values) < 3:
            return "insufficient_data"
        
        try:
            # Simple linear regression
            n = len(values)
            sum_x = sum(days)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(days, values))
            sum_x2 = sum(x * x for x in days)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            if slope > 0.01:
                return "increasing"
            elif slope < -0.01:
                return "decreasing"
            else:
                return "stable"
        except:
            return "unknown"
    
    def _calculate_overall_drift_score(self, drift_analysis: Dict) -> float:
        """Calculate overall drift score from all components."""
        scores = []
        
        # Performance drift
        perf_drift = drift_analysis.get('performance_drift', {})
        if perf_drift.get('detected', False):
            scores.append(perf_drift.get('score', 0))
        
        # Behavioral drift
        behav_drift = drift_analysis.get('behavioral_drift', {})
        if behav_drift.get('detected', False):
            scores.append(behav_drift.get('score', 0))
        
        # Context relevance drift
        context_drift = drift_analysis.get('context_relevance_drift', {})
        if context_drift.get('detected', False):
            scores.append(context_drift.get('score', 0))
        
        # Accuracy drift
        accuracy_drift = drift_analysis.get('accuracy_drift', {})
        if accuracy_drift.get('detected', False):
            scores.append(accuracy_drift.get('score', 0))
        
        if scores:
            return statistics.mean(scores)
        return 0.0
    
    def _determine_drift_severity(self, drift_score: float) -> str:
        """Determine drift severity based on score."""
        if drift_score < 0.05:
            return "none"
        elif drift_score < 0.1:
            return "low"
        elif drift_score < 0.2:
            return "medium"
        elif drift_score < 0.3:
            return "high"
        else:
            return "critical"
    
    def _generate_drift_recommendations(self, drift_analysis: Dict) -> List[str]:
        """Generate recommendations based on drift analysis."""
        recommendations = []
        
        # Performance recommendations
        perf_drift = drift_analysis.get('performance_drift', {})
        if perf_drift.get('detected', False):
            recommendations.append("Consider optimizing response generation pipeline")
            recommendations.append("Review and optimize embedding generation process")
        
        # Behavioral recommendations
        behav_drift = drift_analysis.get('behavioral_drift', {})
        if behav_drift.get('detected', False):
            recommendations.append("Update context retrieval algorithms for new user patterns")
            recommendations.append("Consider retraining semantic models on recent data")
        
        # Context relevance recommendations
        context_drift = drift_analysis.get('context_relevance_drift', {})
        if context_drift.get('detected', False):
            recommendations.append("Review context similarity thresholds")
            recommendations.append("Consider updating semantic embedding models")
        
        # Accuracy recommendations
        accuracy_drift = drift_analysis.get('accuracy_drift', {})
        if accuracy_drift.get('detected', False):
            recommendations.append("Review response generation quality")
            recommendations.append("Consider implementing additional quality checks")
        
        if not recommendations:
            recommendations.append("System performance is stable, continue monitoring")
        
        return recommendations
    
    def _analyze_drift_trends(self, historical_data: Dict) -> Dict:
        """Analyze trends in historical data."""
        trends = {
            'performance_trend': 'stable',
            'usage_trend': 'stable',
            'quality_trend': 'stable'
        }
        
        try:
            performance_metrics = historical_data.get('performance_metrics', [])
            if len(performance_metrics) >= 7:
                response_times = [m.get('response_time', 0) for m in performance_metrics]
                days = list(range(len(response_times)))
                trends['performance_trend'] = self._calculate_trend(response_times, days)
            
            context_usage = historical_data.get('context_usage', [])
            if len(context_usage) >= 7:
                usage_counts = [c.get('usage_count', 0) for c in context_usage]
                days = list(range(len(usage_counts)))
                trends['usage_trend'] = self._calculate_trend(usage_counts, days)
            
            accuracy_metrics = historical_data.get('accuracy_metrics', [])
            if len(accuracy_metrics) >= 7:
                quality_scores = [a.get('response_quality', 0) for a in accuracy_metrics]
                days = list(range(len(quality_scores)))
                trends['quality_trend'] = self._calculate_trend(quality_scores, days)
        
        except Exception as e:
            print(f"⚠️ Error analyzing trends: {e}")
        
        return trends
    
    def _generate_drift_alerts(self, drift_analysis: Dict) -> List[str]:
        """Generate alerts based on drift analysis."""
        alerts = []
        
        severity = drift_analysis.get('drift_severity', 'none')
        if severity in ['high', 'critical']:
            alerts.append(f"🚨 Critical drift detected: {severity} severity")
        
        if drift_analysis.get('drift_detected', False):
            alerts.append("⚠️ System drift detected - review recommendations")
        
        return alerts
    
    def get_drift_summary(self, user_id: str) -> Dict:
        """Get a summary of drift detection results."""
        drift_analysis = self.detect_semantic_drift(user_id)
        
        return {
            'user_id': user_id,
            'drift_detected': drift_analysis.get('drift_detected', False),
            'severity': drift_analysis.get('drift_severity', 'none'),
            'overall_score': drift_analysis.get('overall_drift_score', 0.0),
            'components_affected': [
                component for component, data in drift_analysis.items()
                if isinstance(data, dict) and data.get('detected', False)
            ],
            'recommendations_count': len(drift_analysis.get('recommendations', [])),
            'alerts_count': len(drift_analysis.get('alerts', []))
        }
    
    def set_drift_threshold(self, threshold: float):
        """Set the drift detection threshold."""
        self.drift_threshold = max(0.0, min(1.0, threshold))
        print(f"🔧 Drift threshold set to: {self.drift_threshold}")
    
    def enable_component(self, component: str, enabled: bool = True):
        """Enable or disable drift detection components."""
        if component == 'performance':
            self.performance_monitoring_enabled = enabled
        elif component == 'behavioral':
            self.behavioral_drift_enabled = enabled
        elif component == 'context_relevance':
            self.context_relevance_drift_enabled = enabled
        elif component == 'accuracy':
            self.accuracy_drift_enabled = enabled
        
        print(f"🔧 {component} drift detection {'enabled' if enabled else 'disabled'}")


# Create global instance
semantic_drift_detection = SemanticDriftDetection(r)

# Standalone function for easy import
def detect_semantic_drift(user_id: str, time_window_hours: int = 24) -> Dict:
    """
    Detect semantic drift for a user.
    
    Args:
        user_id: User identifier
        time_window_hours: Time window for analysis (default: 24 hours)
        
    Returns:
        Drift analysis results
    """
    return semantic_drift_detection.detect_semantic_drift(user_id)
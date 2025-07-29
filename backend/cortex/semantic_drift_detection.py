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

# Machine Learning imports
try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ scikit-learn not available, using statistical fallbacks")

class SemanticDriftDetection:
    """
    Detects semantic drift in the context model using ML-enhanced algorithms:
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
        
        # ML models for enhanced drift detection
        if ML_AVAILABLE:
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            self.behavior_clusterer = KMeans(n_clusters=3, random_state=42)
            self.drift_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
            self.scaler = StandardScaler()
            self.pca = PCA(n_components=5)
            self.models_trained = False
        else:
            self.anomaly_detector = None
            self.behavior_clusterer = None
            self.drift_classifier = None
            self.scaler = None
            self.pca = None
            self.models_trained = False
    
    def detect_semantic_drift(self, user_id: str) -> Dict:
        """
        Comprehensive semantic drift detection using ML-enhanced algorithms.
        
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
            'ml_insights': {},
            'recommendations': [],
            'trends': {},
            'alerts': []
        }
        
        # Check cache first
        cache_key = f"drift_analysis:{user_id}"
        cached_data = r.get(cache_key)
        if cached_data:
            try:
                return json.loads(cached_data)
            except json.JSONDecodeError:
                pass
        
        # Get historical data
        historical_data = self._get_historical_data(user_id)
        if not historical_data:
            drift_analysis['alerts'].append("Insufficient historical data for drift detection")
            return drift_analysis
        
        # Train ML models if not trained
        if ML_AVAILABLE and not self.models_trained:
            self._train_ml_models(user_id, historical_data)
        
        # Perform drift detection on each component
        if self.performance_monitoring_enabled:
            drift_analysis['performance_drift'] = self._detect_performance_drift(user_id, historical_data)
        
        if self.behavioral_drift_enabled:
            drift_analysis['behavioral_drift'] = self._detect_behavioral_drift(user_id, historical_data)
        
        if self.context_relevance_drift_enabled:
            drift_analysis['context_relevance_drift'] = self._detect_context_relevance_drift(user_id, historical_data)
        
        if self.accuracy_drift_enabled:
            drift_analysis['accuracy_drift'] = self._detect_accuracy_drift(user_id, historical_data)
        
        # Add ML insights if available
        if ML_AVAILABLE and self.models_trained:
            drift_analysis['ml_insights'] = self._get_ml_insights(user_id, historical_data)
        
        # Calculate overall drift score
        drift_analysis['overall_drift_score'] = self._calculate_overall_drift_score(drift_analysis)
        
        # Determine drift severity
        drift_analysis['drift_detected'] = drift_analysis['overall_drift_score'] > self.drift_threshold
        drift_analysis['drift_severity'] = self._determine_drift_severity(drift_analysis['overall_drift_score'])
        
        # Generate recommendations
        drift_analysis['recommendations'] = self._generate_drift_recommendations(drift_analysis)
        
        # Generate alerts
        drift_analysis['alerts'] = self._generate_drift_alerts(drift_analysis)
        
        # Cache results
        r.setex(cache_key, self.drift_cache_ttl, json.dumps(drift_analysis))
        
        return drift_analysis
    
    def _train_ml_models(self, user_id: str, historical_data: Dict):
        """Train ML models for drift detection."""
        try:
            # Prepare training data
            training_features = self._extract_ml_features(historical_data)
            
            if len(training_features) < self.min_data_points:
                print("⚠️ Insufficient data for ML model training")
                return
            
            # Scale features
            features_scaled = self.scaler.fit_transform(training_features)
            
            # Train anomaly detector
            self.anomaly_detector.fit(features_scaled)
            
            # Train behavior clusterer
            self.behavior_clusterer.fit(features_scaled)
            
            # Prepare drift labels (simplified - would be more sophisticated in production)
            drift_labels = self._generate_drift_labels(historical_data)
            
            # Train drift classifier
            if len(set(drift_labels)) > 1:  # Need at least 2 classes
                self.drift_classifier.fit(features_scaled, drift_labels)
            
            # Apply PCA for dimensionality reduction
            self.pca.fit(features_scaled)
            
            self.models_trained = True
            print(f"✅ ML models trained for drift detection (user: {user_id})")
            
        except Exception as e:
            print(f"❌ ML model training failed: {e}")
            self.models_trained = False
    
    def _extract_ml_features(self, historical_data: Dict) -> List[List[float]]:
        """Extract features for ML models."""
        features = []
        
        # Extract time series data
        timestamps = historical_data.get('timestamps', [])
        response_qualities = historical_data.get('response_qualities', [])
        context_relevances = historical_data.get('context_relevances', [])
        query_lengths = historical_data.get('query_lengths', [])
        response_lengths = historical_data.get('response_lengths', [])
        
        for i in range(len(timestamps)):
            if i < 1:  # Need at least 2 data points for trend calculation
                continue
            
            # Calculate trend features
            quality_trend = response_qualities[i] - response_qualities[i-1] if i > 0 else 0
            relevance_trend = context_relevances[i] - context_relevances[i-1] if i > 0 else 0
            
            # Calculate volatility
            quality_volatility = abs(quality_trend)
            relevance_volatility = abs(relevance_trend)
            
            # Extract features
            feature_vector = [
                response_qualities[i],
                context_relevances[i],
                query_lengths[i] if i < len(query_lengths) else 0,
                response_lengths[i] if i < len(response_lengths) else 0,
                quality_trend,
                relevance_trend,
                quality_volatility,
                relevance_volatility,
                timestamps[i] if timestamps else 0,
                i  # Time index
            ]
            
            features.append(feature_vector)
        
        return features
    
    def _generate_drift_labels(self, historical_data: Dict) -> List[int]:
        """Generate labels for drift classification."""
        labels = []
        
        response_qualities = historical_data.get('response_qualities', [])
        context_relevances = historical_data.get('context_relevances', [])
        
        for i in range(len(response_qualities)):
            if i < 1:
                labels.append(0)  # No drift
                continue
            
            # Calculate drift indicators
            quality_drift = response_qualities[i] - response_qualities[i-1]
            relevance_drift = context_relevances[i] - context_relevances[i-1]
            
            # Determine drift label
            if quality_drift < -0.1 or relevance_drift < -0.1:
                labels.append(1)  # Drift detected
            else:
                labels.append(0)  # No drift
        
        return labels
    
    def _get_ml_insights(self, user_id: str, historical_data: Dict) -> Dict:
        """Get ML-based insights for drift detection."""
        insights = {
            'anomaly_detection': {},
            'behavior_clustering': {},
            'drift_prediction': {},
            'feature_importance': {}
        }
        
        try:
            # Extract current features
            current_features = self._extract_ml_features(historical_data)
            if not current_features:
                return insights
            
            # Get latest features
            latest_features = current_features[-1]
            features_scaled = self.scaler.transform([latest_features])
            
            # Anomaly detection
            anomaly_score = self.anomaly_detector.predict([latest_features])[0]
            anomaly_confidence = self.anomaly_detector.decision_function([latest_features])[0]
            
            insights['anomaly_detection'] = {
                'is_anomaly': bool(anomaly_score == -1),
                'anomaly_confidence': float(anomaly_confidence),
                'anomaly_score': int(anomaly_score)
            }
            
            # Behavior clustering
            cluster_label = self.behavior_clusterer.predict([latest_features])[0]
            cluster_confidence = self.behavior_clusterer.score([latest_features])
            
            insights['behavior_clustering'] = {
                'cluster_label': int(cluster_label),
                'cluster_confidence': float(cluster_confidence),
                'n_clusters': self.behavior_clusterer.n_clusters
            }
            
            # Drift prediction
            if hasattr(self.drift_classifier, 'predict_proba'):
                drift_prob = self.drift_classifier.predict_proba([latest_features])[0]
                insights['drift_prediction'] = {
                    'drift_probability': float(drift_prob[1] if len(drift_prob) > 1 else 0),
                    'no_drift_probability': float(drift_prob[0]),
                    'predicted_drift': bool(self.drift_classifier.predict([latest_features])[0])
                }
            
            # Feature importance (if available)
            if hasattr(self.drift_classifier, 'feature_importances_'):
                insights['feature_importance'] = {
                    'importance_scores': self.drift_classifier.feature_importances_.tolist()
                }
            
            # PCA insights
            if self.pca:
                pca_features = self.pca.transform(features_scaled)
                explained_variance = self.pca.explained_variance_ratio_
                
                insights['pca_analysis'] = {
                    'pca_features': pca_features[0].tolist(),
                    'explained_variance': explained_variance.tolist(),
                    'total_variance_explained': float(sum(explained_variance))
                }
            
        except Exception as e:
            print(f"⚠️ ML insights generation failed: {e}")
        
        return insights
    
    def _get_historical_data(self, user_id: str) -> Dict:
        """
        Retrieve historical performance and behavioral data.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with historical data
        """
        r = self.redis_client
        historical_data = {
            'performance_metrics': [],
            'behavioral_patterns': [],
            'context_usage': [],
            'accuracy_scores': [],
            'timestamps': []
        }
        
        # Get traces with timestamps
        pattern = f"embedding:{user_id}:*"
        trace_keys = r.keys(pattern)
        
        for trace_key in trace_keys:
            trace_data = r.get(trace_key)
            if trace_data:
                try:
                    embedding_data = json.loads(trace_data)
                    metadata = embedding_data.get('metadata', {})
                    created_at = metadata.get('created_at', 0)
                    
                    if created_at > 0:
                        # Calculate days since creation
                        days_old = (time.time() - created_at) / 86400
                        
                        if days_old <= self.window_size:
                            # Performance metrics
                            performance_metrics = {
                                'success_rate': metadata.get('recall_score', 0.0),
                                'quality_score': metadata.get('response_quality', 0.0),
                                'consolidation_score': metadata.get('memory_consolidation_score', 0.0),
                                'precision_score': metadata.get('precision_score', 0.0),
                                'days_old': days_old
                            }
                            historical_data['performance_metrics'].append(performance_metrics)
                            
                            # Behavioral patterns
                            behavioral_patterns = {
                                'complexity': metadata.get('prompt_complexity', 0.0),
                                'semantic_density': metadata.get('semantic_density', 0.0),
                                'conversation_length': metadata.get('conversation_length', 0),
                                'days_old': days_old
                            }
                            historical_data['behavioral_patterns'].append(behavioral_patterns)
                            
                            # Context usage
                            usage_count = r.get(f"usage:{embedding_data.get('trace_id', '')}")
                            context_usage = {
                                'usage_count': int(usage_count) if usage_count else 0,
                                'days_old': days_old
                            }
                            historical_data['context_usage'].append(context_usage)
                            
                            # Accuracy scores (simulated based on metadata)
                            accuracy_score = (
                                metadata.get('recall_score', 0.0) * 0.4 +
                                metadata.get('precision_score', 0.0) * 0.3 +
                                metadata.get('response_quality', 0.0) * 0.3
                            )
                            historical_data['accuracy_scores'].append({
                                'accuracy': accuracy_score,
                                'days_old': days_old
                            })
                            
                            historical_data['timestamps'].append(created_at)
                            
                except json.JSONDecodeError:
                    continue
        
        return historical_data
    
    def _detect_performance_drift(self, user_id: str, historical_data: Dict) -> Dict:
        """
        Detect performance drift over time.
        
        Args:
            user_id: User identifier
            historical_data: Historical performance data
            
        Returns:
            Dict with performance drift analysis
        """
        performance_drift = {
            'drift_detected': False,
            'drift_score': 0.0,
            'trend': 'stable',
            'metrics_affected': [],
            'recommendations': []
        }
        
        metrics = historical_data['performance_metrics']
        if len(metrics) < self.min_data_points:
            return performance_drift
        
        # Analyze each performance metric
        metric_drifts = {}
        
        for metric_name in ['success_rate', 'quality_score', 'consolidation_score', 'precision_score']:
            values = [m[metric_name] for m in metrics]
            days = [m['days_old'] for m in metrics]
            
            if len(values) >= 2:
                # Calculate trend (simple linear regression)
                trend = self._calculate_trend(values, days)
                recent_avg = sum(values[:len(values)//3]) / (len(values)//3)  # Recent 1/3
                older_avg = sum(values[-len(values)//3:]) / (len(values)//3)  # Older 1/3
                
                drift = (older_avg - recent_avg) / max(older_avg, 0.1)  # Normalized drift
                metric_drifts[metric_name] = {
                    'drift': drift,
                    'trend': trend,
                    'recent_avg': recent_avg,
                    'older_avg': older_avg
                }
        
        # Calculate overall performance drift
        if metric_drifts:
            avg_drift = sum(abs(m['drift']) for m in metric_drifts.values()) / len(metric_drifts)
            performance_drift['drift_score'] = avg_drift
            performance_drift['drift_detected'] = avg_drift > self.drift_threshold
            
            # Identify affected metrics
            for metric_name, drift_info in metric_drifts.items():
                if abs(drift_info['drift']) > self.drift_threshold:
                    performance_drift['metrics_affected'].append(metric_name)
            
            # Determine overall trend
            positive_drifts = sum(1 for m in metric_drifts.values() if m['drift'] > 0)
            negative_drifts = sum(1 for m in metric_drifts.values() if m['drift'] < 0)
            
            if positive_drifts > negative_drifts:
                performance_drift['trend'] = 'improving'
            elif negative_drifts > positive_drifts:
                performance_drift['trend'] = 'declining'
            else:
                performance_drift['trend'] = 'mixed'
        
        return performance_drift
    
    def _detect_behavioral_drift(self, user_id: str, historical_data: Dict) -> Dict:
        """
        Detect behavioral pattern drift over time.
        
        Args:
            user_id: User identifier
            historical_data: Historical behavioral data
            
        Returns:
            Dict with behavioral drift analysis
        """
        behavioral_drift = {
            'drift_detected': False,
            'drift_score': 0.0,
            'pattern_changes': [],
            'learning_curve_shift': False,
            'complexity_evolution': 'stable'
        }
        
        patterns = historical_data['behavioral_patterns']
        if len(patterns) < self.min_data_points:
            return behavioral_drift
        
        # Analyze complexity evolution
        complexities = [p['complexity'] for p in patterns]
        days = [p['days_old'] for p in patterns]
        
        if len(complexities) >= 2:
            complexity_trend = self._calculate_trend(complexities, days)
            recent_complexity = sum(complexities[:len(complexities)//3]) / (len(complexities)//3)
            older_complexity = sum(complexities[-len(complexities)//3:]) / (len(complexities)//3)
            
            complexity_drift = (recent_complexity - older_complexity) / max(older_complexity, 0.1)
            behavioral_drift['drift_score'] = abs(complexity_drift)
            behavioral_drift['drift_detected'] = abs(complexity_drift) > self.drift_threshold
            
            # Determine complexity evolution
            if complexity_drift > 0.1:
                behavioral_drift['complexity_evolution'] = 'increasing'
                behavioral_drift['pattern_changes'].append('User queries becoming more complex')
            elif complexity_drift < -0.1:
                behavioral_drift['complexity_evolution'] = 'decreasing'
                behavioral_drift['pattern_changes'].append('User queries becoming simpler')
            
            # Check for learning curve shifts
            if abs(complexity_drift) > 0.2:
                behavioral_drift['learning_curve_shift'] = True
                behavioral_drift['pattern_changes'].append('Significant learning curve shift detected')
        
        # Analyze semantic density changes
        densities = [p['semantic_density'] for p in patterns]
        if len(densities) >= 2:
            recent_density = sum(densities[:len(densities)//3]) / (len(densities)//3)
            older_density = sum(densities[-len(densities)//3:]) / (len(densities)//3)
            
            density_drift = (recent_density - older_density) / max(older_density, 0.1)
            if abs(density_drift) > self.drift_threshold:
                behavioral_drift['pattern_changes'].append('Semantic density pattern changed')
        
        return behavioral_drift
    
    def _detect_context_relevance_drift(self, user_id: str, historical_data: Dict) -> Dict:
        """
        Detect context relevance drift over time.
        
        Args:
            user_id: User identifier
            historical_data: Historical context usage data
            
        Returns:
            Dict with context relevance drift analysis
        """
        context_drift = {
            'drift_detected': False,
            'drift_score': 0.0,
            'usage_pattern_changes': [],
            'relevance_degradation': False,
            'context_efficiency': 'stable'
        }
        
        usage_data = historical_data['context_usage']
        if len(usage_data) < self.min_data_points:
            return context_drift
        
        # Analyze usage patterns
        usage_counts = [u['usage_count'] for u in usage_data]
        days = [u['days_old'] for u in usage_data]
        
        if len(usage_counts) >= 2:
            usage_trend = self._calculate_trend(usage_counts, days)
            recent_usage = sum(usage_counts[:len(usage_counts)//3]) / (len(usage_counts)//3)
            older_usage = sum(usage_counts[-len(usage_counts)//3:]) / (len(usage_counts)//3)
            
            usage_drift = (older_usage - recent_usage) / max(older_usage, 1)
            context_drift['drift_score'] = abs(usage_drift)
            context_drift['drift_detected'] = abs(usage_drift) > self.drift_threshold
            
            # Determine context efficiency
            if usage_drift > 0.2:
                context_drift['context_efficiency'] = 'declining'
                context_drift['usage_pattern_changes'].append('Context usage decreasing')
                context_drift['relevance_degradation'] = True
            elif usage_drift < -0.2:
                context_drift['context_efficiency'] = 'improving'
                context_drift['usage_pattern_changes'].append('Context usage increasing')
            
            # Check for relevance degradation
            if recent_usage < older_usage * 0.5:  # 50% drop in usage
                context_drift['relevance_degradation'] = True
                context_drift['usage_pattern_changes'].append('Significant context relevance drop')
        
        return context_drift
    
    def _detect_accuracy_drift(self, user_id: str, historical_data: Dict) -> Dict:
        """
        Detect accuracy drift over time.
        
        Args:
            user_id: User identifier
            historical_data: Historical accuracy data
            
        Returns:
            Dict with accuracy drift analysis
        """
        accuracy_drift = {
            'drift_detected': False,
            'drift_score': 0.0,
            'accuracy_trend': 'stable',
            'confidence_degradation': False,
            'prediction_quality': 'stable'
        }
        
        accuracy_scores = historical_data['accuracy_scores']
        if len(accuracy_scores) < self.min_data_points:
            return accuracy_drift
        
        # Analyze accuracy trends
        accuracies = [a['accuracy'] for a in accuracy_scores]
        days = [a['days_old'] for a in accuracy_scores]
        
        if len(accuracies) >= 2:
            accuracy_trend = self._calculate_trend(accuracies, days)
            recent_accuracy = sum(accuracies[:len(accuracies)//3]) / (len(accuracies)//3)
            older_accuracy = sum(accuracies[-len(accuracies)//3:]) / (len(accuracies)//3)
            
            accuracy_drift_value = (older_accuracy - recent_accuracy) / max(older_accuracy, 0.1)
            accuracy_drift['drift_score'] = abs(accuracy_drift_value)
            accuracy_drift['drift_detected'] = abs(accuracy_drift_value) > self.drift_threshold
            
            # Determine accuracy trend
            if accuracy_drift_value > 0.1:
                accuracy_drift['accuracy_trend'] = 'declining'
                accuracy_drift['confidence_degradation'] = True
            elif accuracy_drift_value < -0.1:
                accuracy_drift['accuracy_trend'] = 'improving'
            
            # Check for significant degradation
            if recent_accuracy < older_accuracy * 0.7:  # 30% drop
                accuracy_drift['confidence_degradation'] = True
                accuracy_drift['prediction_quality'] = 'degraded'
        
        return accuracy_drift
    
    def _calculate_trend(self, values: List[float], days: List[float]) -> str:
        """
        Calculate trend direction for a series of values.
        
        Args:
            values: List of values
            days: List of days (timestamps)
            
        Returns:
            Trend direction: 'increasing', 'decreasing', or 'stable'
        """
        if len(values) < 2:
            return 'stable'
        
        # Simple linear trend calculation
        recent_values = values[:len(values)//3]
        older_values = values[-len(values)//3:]
        
        recent_avg = sum(recent_values) / len(recent_values)
        older_avg = sum(older_values) / len(older_values)
        
        change = recent_avg - older_avg
        threshold = max(older_avg * 0.1, 0.05)  # 10% change threshold
        
        if change > threshold:
            return 'increasing'
        elif change < -threshold:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_overall_drift_score(self, drift_analysis: Dict) -> float:
        """
        Calculate overall drift score from all components.
        
        Args:
            drift_analysis: Complete drift analysis
            
        Returns:
            Overall drift score (0-1)
        """
        component_scores = []
        
        # Performance drift
        if drift_analysis['performance_drift'].get('drift_score', 0) > 0:
            component_scores.append(drift_analysis['performance_drift']['drift_score'])
        
        # Behavioral drift
        if drift_analysis['behavioral_drift'].get('drift_score', 0) > 0:
            component_scores.append(drift_analysis['behavioral_drift']['drift_score'])
        
        # Context relevance drift
        if drift_analysis['context_relevance_drift'].get('drift_score', 0) > 0:
            component_scores.append(drift_analysis['context_relevance_drift']['drift_score'])
        
        # Accuracy drift
        if drift_analysis['accuracy_drift'].get('drift_score', 0) > 0:
            component_scores.append(drift_analysis['accuracy_drift']['drift_score'])
        
        if not component_scores:
            return 0.0
        
        # Weighted average (performance and accuracy weighted higher)
        weights = [0.3, 0.2, 0.2, 0.3]  # performance, behavioral, context, accuracy
        weighted_sum = sum(score * weight for score, weight in zip(component_scores, weights))
        
        return min(weighted_sum, 1.0)
    
    def _determine_drift_severity(self, drift_score: float) -> str:
        """
        Determine drift severity based on score.
        
        Args:
            drift_score: Overall drift score
            
        Returns:
            Severity level: 'none', 'low', 'medium', 'high', 'critical'
        """
        if drift_score < 0.1:
            return 'none'
        elif drift_score < 0.2:
            return 'low'
        elif drift_score < 0.35:
            return 'medium'
        elif drift_score < 0.5:
            return 'high'
        else:
            return 'critical'
    
    def _generate_drift_recommendations(self, drift_analysis: Dict) -> List[str]:
        """
        Generate recommendations based on drift analysis.
        
        Args:
            drift_analysis: Complete drift analysis
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if drift_analysis['drift_severity'] == 'critical':
            recommendations.append("CRITICAL: Immediate model retraining required")
            recommendations.append("Consider reducing drift threshold for early detection")
        
        if drift_analysis['drift_severity'] in ['high', 'critical']:
            recommendations.append("Increase monitoring frequency")
            recommendations.append("Review and update context relevance criteria")
        
        # Performance-specific recommendations
        performance_drift = drift_analysis['performance_drift']
        if performance_drift.get('drift_detected', False):
            if 'success_rate' in performance_drift.get('metrics_affected', []):
                recommendations.append("Optimize context selection criteria")
            if 'quality_score' in performance_drift.get('metrics_affected', []):
                recommendations.append("Review response quality assessment metrics")
        
        # Behavioral-specific recommendations
        behavioral_drift = drift_analysis['behavioral_drift']
        if behavioral_drift.get('learning_curve_shift', False):
            recommendations.append("Update user expertise level detection")
            recommendations.append("Adjust complexity scoring algorithms")
        
        # Context-specific recommendations
        context_drift = drift_analysis['context_relevance_drift']
        if context_drift.get('relevance_degradation', False):
            recommendations.append("Refresh context embeddings")
            recommendations.append("Update semantic similarity thresholds")
        
        # Accuracy-specific recommendations
        accuracy_drift = drift_analysis['accuracy_drift']
        if accuracy_drift.get('confidence_degradation', False):
            recommendations.append("Retrain semantic embedding model")
            recommendations.append("Update prediction confidence thresholds")
        
        if not recommendations:
            recommendations.append("System performing within normal parameters")
        
        return recommendations
    
    def _analyze_drift_trends(self, historical_data: Dict) -> Dict:
        """
        Analyze long-term drift trends.
        
        Args:
            historical_data: Historical data
            
        Returns:
            Dict with trend analysis
        """
        trends = {
            'overall_trend': 'stable',
            'trend_duration_days': 0,
            'trend_velocity': 0.0,
            'seasonal_patterns': False,
            'volatility': 'low'
        }
        
        # Calculate trend velocity (rate of change)
        if len(historical_data['timestamps']) >= 2:
            timestamps = sorted(historical_data['timestamps'])
            duration_days = (max(timestamps) - min(timestamps)) / 86400
            trends['trend_duration_days'] = duration_days
            
            # Calculate volatility
            performance_values = [m['success_rate'] for m in historical_data['performance_metrics']]
            if len(performance_values) >= 2:
                variance = sum((x - sum(performance_values)/len(performance_values))**2 for x in performance_values) / len(performance_values)
                std_dev = variance ** 0.5
                
                if std_dev > 0.2:
                    trends['volatility'] = 'high'
                elif std_dev > 0.1:
                    trends['volatility'] = 'medium'
                else:
                    trends['volatility'] = 'low'
        
        return trends
    
    def _generate_drift_alerts(self, drift_analysis: Dict) -> List[str]:
        """
        Generate alerts based on drift analysis.
        
        Args:
            drift_analysis: Complete drift analysis
            
        Returns:
            List of alerts
        """
        alerts = []
        
        if drift_analysis['drift_severity'] == 'critical':
            alerts.append("🚨 CRITICAL DRIFT DETECTED: System performance severely degraded")
        
        if drift_analysis['drift_severity'] == 'high':
            alerts.append("⚠️ HIGH DRIFT DETECTED: Performance monitoring required")
        
        # Component-specific alerts
        if drift_analysis['performance_drift'].get('drift_detected', False):
            alerts.append("📉 Performance drift detected in context model")
        
        if drift_analysis['behavioral_drift'].get('learning_curve_shift', False):
            alerts.append("🔄 User behavior pattern shift detected")
        
        if drift_analysis['context_relevance_drift'].get('relevance_degradation', False):
            alerts.append("🔍 Context relevance degradation detected")
        
        if drift_analysis['accuracy_drift'].get('confidence_degradation', False):
            alerts.append("🎯 Model accuracy degradation detected")
        
        return alerts
    
    def get_drift_summary(self, user_id: str) -> Dict:
        """
        Get a summary of drift detection results.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with drift summary
        """
        drift_analysis = self.detect_semantic_drift(user_id)
        
        summary = {
            'user_id': user_id,
            'drift_status': 'stable' if not drift_analysis['drift_detected'] else 'drift_detected',
            'severity': drift_analysis['drift_severity'],
            'overall_score': drift_analysis['overall_drift_score'],
            'components_affected': [],
            'recommendations_count': len(drift_analysis['recommendations']),
            'alerts_count': len(drift_analysis['alerts']),
            'last_updated': time.time()
        }
        
        # Identify affected components
        if drift_analysis['performance_drift'].get('drift_detected', False):
            summary['components_affected'].append('performance')
        
        if drift_analysis['behavioral_drift'].get('drift_detected', False):
            summary['components_affected'].append('behavioral')
        
        if drift_analysis['context_relevance_drift'].get('drift_detected', False):
            summary['components_affected'].append('context_relevance')
        
        if drift_analysis['accuracy_drift'].get('drift_detected', False):
            summary['components_affected'].append('accuracy')
        
        return summary
    
    def set_drift_threshold(self, threshold: float):
        """
        Set custom drift detection threshold.
        
        Args:
            threshold: New threshold value (0-1)
        """
        if 0 <= threshold <= 1:
            self.drift_threshold = threshold
        else:
            raise ValueError("Drift threshold must be between 0 and 1")
    
    def enable_component(self, component: str, enabled: bool = True):
        """
        Enable or disable drift detection components.
        
        Args:
            component: Component name ('performance', 'behavioral', 'context_relevance', 'accuracy')
            enabled: Whether to enable the component
        """
        if component == 'performance':
            self.performance_monitoring_enabled = enabled
        elif component == 'behavioral':
            self.behavioral_drift_enabled = enabled
        elif component == 'context_relevance':
            self.context_relevance_drift_enabled = enabled
        elif component == 'accuracy':
            self.accuracy_drift_enabled = enabled
        else:
            raise ValueError(f"Unknown component: {component}")
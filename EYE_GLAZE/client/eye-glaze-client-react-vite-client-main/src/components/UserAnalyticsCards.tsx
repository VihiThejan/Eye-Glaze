import { useEffect, useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import { Eye, TrendingUp, Clock, Target, RefreshCw } from 'lucide-react';
import { formatDistanceToNow, format } from 'date-fns';
import { User } from '@/types';

interface AnalysisData {
  _id: string;
  username: string;
  hasStress: boolean;
  imageUrl: string | null;
  createdAt: string;
}

interface UserAnalytics {
  count: number;
  data: AnalysisData[];
}

interface LatestAnalysis {
  _id: string;
  username: string;
  hasStress: boolean;
  imageUrl: string | null;
  createdAt: string;
}

interface WeeklyTrend {
  week: string;
  total: number;
  stressDetected: number;
  percentage: number;
}

interface HealthSummary {
  summary: {
    totalAnalyses: number;
    stressDetectedCount: number;
    stressPercentage: number;
    latestStatus: string;
    latestAnalysisTime: string;
  };
  weeklyTrends: WeeklyTrend[];
}

interface StatCard {
  icon: typeof Eye;
  title: string;
  value: string;
  description: string;
  color: string;
  trend: string;
  extraData?: WeeklyTrend[];
}

export function UserAnalyticsCards() {
  const [analytics, setAnalytics] = useState<UserAnalytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [user, setUser] = useState<User | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [latestAnalysis, setLatestAnalysis] = useState<LatestAnalysis | null>(null);
  const [healthSummary, setHealthSummary] = useState<HealthSummary | null>(null);
  
  // Get user from localStorage directly - simplified version
  useEffect(() => {
    const getUserFromStorage = (): User | null => {
      const storedUser = localStorage.getItem('eyeGlazeUser');
      return storedUser ? JSON.parse(storedUser) : null;
    };
    
    // Set initial user - just once
    setUser(getUserFromStorage());
    
    // No intervals or event listeners to prevent rerendering loops
  }, []);

  // Define fetchUserAnalytics as a callback to allow manual refreshing - simplified
  const fetchUserAnalytics = useCallback(async () => {
    if (!user) return;
    
    // Determine which user identifier to use - try email first (likely username), then name
    const userIdentifier = user.email || user.name;
    
    if (!userIdentifier) return;

    setLoading(true);
    setRefreshing(true);
    
    try {
      // Use email as username for the API call since that's what we use for login
      const response = await fetch(`http://localhost:5176/api/analysis/user/${userIdentifier}`);

      if (!response.ok) {
        throw new Error(`Failed to fetch user analytics: ${response.status}`);
      }

      const data = await response.json();
      setAnalytics(data);
      
      // Only show toast on manual refresh button click (initiated by user)
      if (refreshing && !loading) {
        toast.success('Analytics data refreshed');
      }
    } catch (error) {
      // Only show error toast on manual refresh (to avoid annoying the user)
      if (refreshing && !loading) {
        toast.error('Failed to load analytics data');
      }
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [user]);

  // Function to fetch the latest analysis for the user
  const fetchLatestAnalysis = useCallback(async () => {
    if (!user) return;
    
    // Use email as identifier for API call
    const userEmail = user.email;
    
    if (!userEmail) return;
    
    try {
      const response = await fetch(`http://localhost:5176/api/analysis/latest/${userEmail}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch latest analysis: ${response.status}`);
      }
      
      const result = await response.json();
      if (result.status === "success" && result.data) {
        setLatestAnalysis(result.data);
      }
    } catch (error) {
      // Only log error, don't show toast to avoid interrupting user
      console.error("Error fetching latest analysis:", error);
    }
  }, [user]);

  // Function to fetch health summary data
  const fetchHealthSummary = useCallback(async () => {
    if (!user) return;
    
    // Use email as identifier for API call
    const userEmail = user.email;
    
    if (!userEmail) return;
    
    try {
      const response = await fetch(`http://localhost:5176/api/recommendations/summary/${userEmail}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch health summary: ${response.status}`);
      }
      
      const result = await response.json();
      if (result.status === "success" && result.data) {
        setHealthSummary(result.data);
      }
    } catch (error) {
      console.error("Error fetching health summary:", error);
    }
  }, [user]);

  // Fetch analytics, latest analysis, and health summary when user changes
  useEffect(() => {
    if (user) {
      fetchUserAnalytics();
      fetchLatestAnalysis();
      fetchHealthSummary();
    }
  }, [user, fetchUserAnalytics, fetchLatestAnalysis, fetchHealthSummary]);

  // Calculate statistics from analytics data - using health summary when available
  const calculateStats = (): StatCard[] => {
    // Prefer health summary data if available
    if (healthSummary) {
      const { summary, weeklyTrends } = healthSummary;
      
      // Calculate most recent analysis time
      const latestAnalysisTime = latestAnalysis?.createdAt 
        ? formatDistanceToNow(new Date(latestAnalysis.createdAt), { addSuffix: true })
        : summary.latestAnalysisTime 
          ? formatDistanceToNow(new Date(summary.latestAnalysisTime), { addSuffix: true })
          : 'No Data';
      
      // Calculate wellness goal progress (inverse of stress percentage)
      const wellnessGoal = Math.max(0, 100 - summary.stressPercentage);
      
      return [
        {
          icon: Eye,
          title: 'Total Analyses',
          value: summary.totalAnalyses.toString(),
          description: 'Eye scans completed',
          color: 'text-green-500',
          trend: 'New',
        },
        {
          icon: TrendingUp,
          title: 'Average Stress',
          value: `${Math.round(summary.stressPercentage / 10)}/10`,
          description: 'Based on all scans',
          color: 'text-emerald-400',
          trend: '+7%',
        },
        {
          icon: Clock,
          title: 'Last Analysis',
          value: latestAnalysisTime,
          description: summary.latestStatus || 'No recent scans',
          color: summary.latestStatus?.toLowerCase().includes('stress') 
            ? 'text-red-400' 
            : 'text-green-400',
          trend: 'Recent',
        },
        {
          icon: Target,
          title: 'Wellness Goal',
          value: `${wellnessGoal}%`,
          description: 'Progress to target',
          color: 'text-yellow-400',
          trend: '-3%',
          extraData: weeklyTrends,
        },
      ];
    }
    
    // Fallback to analytics data if health summary is not available
    if (!analytics || !analytics.data || analytics.data.length === 0) {
      return defaultStats;
    }
    
    const totalAnalyses = analytics.count;
    
    // Calculate average stress (percentage of stress detections)
    const stressCount = analytics.data.filter(item => item.hasStress).length;
    
    const stressPercentage = totalAnalyses > 0 
      ? Math.round((stressCount / totalAnalyses) * 10)
      : 0;
    
    // Get most recent analysis time
    const latestAnalysisTime = latestAnalysis?.createdAt 
      ? formatDistanceToNow(new Date(latestAnalysis.createdAt), { addSuffix: true })
      : analytics.data[0]?.createdAt 
        ? formatDistanceToNow(new Date(analytics.data[0].createdAt), { addSuffix: true })
        : 'No Data';
    
    // Wellness goal progress (inverse of stress percentage)
    const wellnessGoal = Math.max(0, 100 - (stressCount / totalAnalyses * 100) || 0);
    
    return [
      {
        icon: Eye,
        title: 'Total Analyses',
        value: totalAnalyses.toString(),
        description: 'Eye scans completed',
        color: 'text-green-500',
        trend: 'New',
      },
      {
        icon: TrendingUp,
        title: 'Average Stress',
        value: `${stressPercentage}/10`,
        description: 'Based on all scans',
        color: 'text-emerald-400',
        trend: '+7%',
      },
      {
        icon: Clock,
        title: 'Last Analysis',
        value: latestAnalysisTime,
        description: latestAnalysis?.hasStress !== undefined
          ? (latestAnalysis.hasStress ? 'Stress detected' : 'No stress detected')
          : 'No recent scans',
        color: latestAnalysis?.hasStress !== undefined
          ? (latestAnalysis.hasStress ? 'text-red-400' : 'text-green-400')
          : 'text-blue-400',
        trend: 'Recent',
      },
      {
        icon: Target,
        title: 'Wellness Goal',
        value: `${Math.round(wellnessGoal)}%`,
        description: 'Progress to target',
        color: 'text-yellow-400',
        trend: '-3%',
        extraData: [],
      },
    ];
  };

  // Default stats to show when loading or no data
  const defaultStats: StatCard[] = [
    {
      icon: Eye,
      title: 'Total Analyses',
      value: '0',
      description: 'Eye scans completed',
      color: 'text-green-500',
      trend: 'New',
    },
    {
      icon: TrendingUp,
      title: 'Average Stress',
      value: '0/10',
      description: 'Based on all scans',
      color: 'text-accent',
      trend: '—',
    },
    {
      icon: Clock,
      title: 'Last Analysis',
      value: 'No Data',
      description: 'No recent scans',
      color: 'text-blue-400',
      trend: '—',
    },
    {
      icon: Target,
      title: 'Wellness Goal',
      value: '0%',
      description: 'Progress to target',
      color: 'text-yellow-400',
      trend: '—',
      extraData: [],
    },
  ];

  const stats = loading ? defaultStats : calculateStats();

  // Function to manually refresh data
  const handleRefresh = () => {
    setRefreshing(true);
    fetchUserAnalytics();
    fetchLatestAnalysis();
    fetchHealthSummary();
  };

  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">Your Analytics</h2>
        <Button 
          variant="outline" 
          size="sm" 
          onClick={handleRefresh} 
          disabled={loading || refreshing}
          className="flex items-center gap-2 px-4 py-2 rounded-full"
        >
          <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
          <span>{refreshing ? 'Refreshing...' : 'Refresh Data'}</span>
        </Button>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {stats.map((stat, index) => (
          <Card key={index} className="glass-card hover:glow-primary transition-all duration-300 dark:bg-navy-800 overflow-hidden border-none min-h-[180px] flex flex-col">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 py-3 pb-0">
              <div className="flex items-center gap-2">
                <stat.icon className={`h-5 w-5 ${stat.color}`} />
                <CardTitle className="text-base font-semibold">{stat.title}</CardTitle>
              </div>
              {stat.trend !== '—' && (
                <Badge 
                  variant={stat.trend.includes('-') ? "destructive" : "outline"} 
                  className={`text-xs px-2 py-0.5 ${stat.trend.includes('-') ? 'bg-red-900/20 text-red-400' : 'bg-emerald-900/20 text-emerald-400'}`}
                >
                  {stat.trend}
                </Badge>
              )}
            </CardHeader>
            
            <CardContent className="pt-2 pb-4 flex-grow flex flex-col">
              {/* Special handling for Last Analysis card to show image preview if available */}
              {stat.title === "Last Analysis" && latestAnalysis?.imageUrl && (
                <div className="mb-2 relative rounded-lg overflow-hidden" style={{ height: "100px" }}>
                  <img 
                    src={latestAnalysis.imageUrl} 
                    alt="Eye scan" 
                    className="absolute inset-0 w-full h-full object-cover"
                  />
                  <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
                    <div className={`text-sm py-1 px-3 ${latestAnalysis.hasStress ? 'bg-red-700 text-white' : 'bg-emerald-700 text-white'} rounded-md font-medium`}>
                      {latestAnalysis.hasStress ? "Stress Detected" : "No Stress"}
                    </div>
                  </div>
                </div>
              )}
              
              {stat.title === "Last Analysis" && latestAnalysis?.createdAt ? (
                <>
                  <div className={`text-3xl font-bold ${latestAnalysis.hasStress ? 'text-red-400' : 'text-green-400'} mb-1`}>
                    {formatDistanceToNow(new Date(latestAnalysis.createdAt), { addSuffix: true })}
                  </div>
                  <p className="text-sm text-muted-foreground mb-auto">{stat.description}</p>
                  <p className="text-xs text-muted-foreground mt-2 opacity-80">
                    {format(new Date(latestAnalysis.createdAt), "MMM d, yyyy 'at' h:mm a")}
                  </p>
                </>
              ) : stat.title === "Wellness Goal" && stat.extraData && stat.extraData.length > 0 ? (
                <>
                  <div className={`text-3xl font-bold ${stat.color} mb-1`}>{stat.value}</div>
                  <p className="text-sm text-muted-foreground mb-2">{stat.description}</p>
                  <div className="mt-2 space-y-1">
                    {stat.extraData.map((week, i) => (
                      <div key={i} className="flex flex-col">
                        <div className="flex justify-between items-center text-xs text-muted-foreground">
                          <span>Week of {format(new Date(week.week), "MMM d")}</span>
                          <span className={week.percentage > 50 ? "text-red-400" : "text-green-400"}>
                            {week.stressDetected}/{week.total} scans
                          </span>
                        </div>
                        <div className="w-full h-1.5 bg-gray-700 rounded-full overflow-hidden mt-1">
                          <div 
                            className={`h-full ${week.percentage > 50 ? "bg-red-500" : "bg-green-500"}`} 
                            style={{ width: `${week.percentage}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <>
                  <div className={`text-3xl font-bold ${stat.color} mb-1`}>{stat.value}</div>
                  <p className="text-sm text-muted-foreground mb-auto">{stat.description}</p>
                </>
              )}
            </CardContent>
          </Card>
        ))}
      </div>


    </>
  );
}
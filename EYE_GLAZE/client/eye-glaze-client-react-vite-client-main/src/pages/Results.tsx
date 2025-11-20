import { useState } from 'react';
import { Calendar, TrendingDown, TrendingUp, AlertTriangle, Target, Award, Clock, Brain, Activity } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, AreaChart, Area } from 'recharts';
import { AnalysisResults } from '@/components/AnalysisResults';
import { StressRecommendations } from '@/components/StressRecommendations';

export function Results() {
  // We'll keep the state but mark it with underscore to indicate it's intentionally unused for now
  const [_selectedPeriod, _setSelectedPeriod] = useState('7d');

  const historicalData = [
    { date: 'Jan 1', stress: 4.2, mood: 6.5, energy: 7.2 },
    { date: 'Jan 2', stress: 3.8, mood: 7.0, energy: 7.8 },
    { date: 'Jan 3', stress: 5.1, mood: 5.8, energy: 6.2 },
    { date: 'Jan 4', stress: 2.9, mood: 8.2, energy: 8.5 },
    { date: 'Jan 5', stress: 3.5, mood: 7.5, energy: 7.9 },
    { date: 'Jan 6', stress: 4.0, mood: 6.8, energy: 7.1 },
    { date: 'Jan 7', stress: 3.2, mood: 7.8, energy: 8.0 },
  ];

  const weeklyBreakdown = [
    { day: 'Mon', stress: 4.2, scans: 3, recovery: 85 },
    { day: 'Tue', stress: 3.8, scans: 2, recovery: 78 },
    { day: 'Wed', stress: 5.1, scans: 4, recovery: 65 },
    { day: 'Thu', stress: 2.9, scans: 2, recovery: 92 },
    { day: 'Fri', stress: 3.5, scans: 3, recovery: 80 },
    { day: 'Sat', stress: 4.0, scans: 1, recovery: 75 },
    { day: 'Sun', stress: 3.2, scans: 2, recovery: 88 },
  ];

  return (
    <div className="min-h-screen pt-28 p-4 bg-gradient-to-br from-background via-background to-primary/5">
      <div className="container mx-auto max-w-7xl">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold mb-2">
                <span className="text-gradient">Stress Analysis</span> Results
              </h1>
              <p className="text-lg text-muted-foreground">
                Comprehensive insights into your stress patterns and wellness metrics
              </p>
            </div>
            <Badge variant="secondary" className="px-4 py-2">
              <Calendar className="w-4 h-4 mr-2" />
              Last 7 days
            </Badge>
          </div>
        </div>

        <Tabs defaultValue="current" className="space-y-8">
          <TabsList className="grid w-full grid-cols-3 max-w-md">
            <TabsTrigger value="current">Current</TabsTrigger>
            <TabsTrigger value="trends">Trends</TabsTrigger>
            <TabsTrigger value="insights">Insights</TabsTrigger>
          </TabsList>

          <TabsContent value="current" className="space-y-8">
            {/* Current Reading */}
            <AnalysisResults />

            {/* Recommendations */}
            <StressRecommendations />
          </TabsContent>

          <TabsContent value="trends" className="space-y-8">
            {/* Stress Trend Chart */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2 text-2xl">
                  <TrendingDown className="h-6 w-6" />
                  <span>Stress Level Trends</span>
                </CardTitle>
                <CardDescription className="text-base">
                  Your stress patterns and recovery metrics over time
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-96">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={historicalData}>
                      <defs>
                        <linearGradient id="stressGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis dataKey="date" stroke="hsl(var(--muted-foreground))" />
                      <YAxis stroke="hsl(var(--muted-foreground))" />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: 'hsl(var(--card))',
                          border: '1px solid hsl(var(--border))',
                          borderRadius: '8px',
                          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)'
                        }}
                      />
                      <Area 
                        type="monotone" 
                        dataKey="stress" 
                        stroke="hsl(var(--primary))" 
                        strokeWidth={3}
                        fill="url(#stressGradient)"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Weekly Breakdown */}
            <div className="grid lg:grid-cols-2 gap-8">
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="text-xl">Daily Stress Levels</CardTitle>
                  <CardDescription>Average stress by day of week</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={weeklyBreakdown}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                        <XAxis dataKey="day" stroke="hsl(var(--muted-foreground))" />
                        <YAxis stroke="hsl(var(--muted-foreground))" />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'hsl(var(--card))',
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '8px'
                          }}
                        />
                        <Bar dataKey="stress" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="text-xl">Recovery Patterns</CardTitle>
                  <CardDescription>Stress recovery efficiency</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={weeklyBreakdown}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                        <XAxis dataKey="day" stroke="hsl(var(--muted-foreground))" />
                        <YAxis stroke="hsl(var(--muted-foreground))" />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'hsl(var(--card))',
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '8px'
                          }}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="recovery" 
                          stroke="hsl(var(--accent))" 
                          strokeWidth={3}
                          dot={{ fill: 'hsl(var(--accent))', strokeWidth: 2, r: 4 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="insights" className="space-y-8">
            {/* Insights Cards */}
            <div className="grid lg:grid-cols-2 gap-8">
              <Card className="glass-card border-green-500/20">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2 text-xl text-green-400">
                    <TrendingDown className="h-5 w-5" />
                    <span>Positive Trends</span>
                  </CardTitle>
                  <CardDescription>Areas where you're making great progress</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {[
                    { text: '15% reduction in average stress levels this week', icon: TrendingDown },
                    { text: 'Improved sleep quality detected in recent analyses', icon: Brain },
                    { text: 'Faster stress recovery after implementing breaks', icon: Activity },
                    { text: 'More consistent daily wellness patterns', icon: Target },
                  ].map((item, index) => (
                    <div key={index} className="flex items-center space-x-3">
                      <div className="w-8 h-8 rounded-lg bg-green-400/20 flex items-center justify-center">
                        <item.icon className="w-4 h-4 text-green-400" />
                      </div>
                      <span className="text-sm">{item.text}</span>
                    </div>
                  ))}
                </CardContent>
              </Card>

              <Card className="glass-card border-yellow-500/20">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2 text-xl text-yellow-400">
                    <AlertTriangle className="h-5 w-5" />
                    <span>Areas for Improvement</span>
                  </CardTitle>
                  <CardDescription>Opportunities to optimize your wellness</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {[
                    { text: 'Stress peaks typically occur around 2-3 PM daily', icon: Clock },
                    { text: 'Consider more frequent micro-breaks during work', icon: Target },
                    { text: 'Weekend stress levels could be further optimized', icon: Calendar },
                    { text: 'Hydration levels may impact stress responses', icon: Activity },
                  ].map((item, index) => (
                    <div key={index} className="flex items-center space-x-3">
                      <div className="w-8 h-8 rounded-lg bg-yellow-400/20 flex items-center justify-center">
                        <item.icon className="w-4 h-4 text-yellow-400" />
                      </div>
                      <span className="text-sm">{item.text}</span>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>

            {/* Wellness Score */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2 text-2xl">
                  <Award className="h-6 w-6" />
                  <span>Overall Wellness Score</span>
                </CardTitle>
                <CardDescription className="text-base">
                  Comprehensive assessment based on stress patterns, recovery rates, and improvement trends
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-8">
                  <div className="text-center space-y-4">
                    <div className="text-6xl font-bold text-primary">78</div>
                    <div className="text-lg text-muted-foreground">Wellness Score</div>
                    <Badge variant="secondary" className="px-4 py-2">
                      <TrendingUp className="w-4 h-4 mr-2" />
                      Excellent Progress
                    </Badge>
                  </div>
                  <div className="space-y-6">
                    <div>
                      <Progress value={78} className="h-6 mb-3" />
                      <div className="flex justify-between text-sm text-muted-foreground">
                        <span>Needs Work</span>
                        <span>Good</span>
                        <span>Excellent</span>
                      </div>
                    </div>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-sm">Stress Management</span>
                        <span className="text-sm font-medium">82%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Recovery Rate</span>
                        <span className="text-sm font-medium">75%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Consistency</span>
                        <span className="text-sm font-medium">79%</span>
                      </div>
                    </div>
                  </div>
                </div>
                <Alert className="mt-6 border-primary/20 bg-primary/5">
                  <Award className="h-4 w-4" />
                  <AlertDescription>
                    Outstanding progress! You're in the top 25% of users with similar profiles. 
                    Your consistent approach to stress management is paying off with measurable improvements 
                    in your overall wellness metrics.
                  </AlertDescription>
                </Alert>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}